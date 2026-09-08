import { ProteinMPNN, modelManifestURL } from 'proteinmpnn-web';

let model, key, controller;
self.onmessage = async ({ data }) => {
  if (data.type === 'cancel') { controller?.abort(); return; }
  if (data.type !== 'sample' || controller) return;
  controller = new AbortController();
  const signal = controller.signal, id = data.id;
  const send = value => self.postMessage({ id, ...value });
  try {
    const { settings, input, options, count, seed, wasmPaths } = data;
    const wanted = `${settings.family}/${settings.model}/${settings.backend}/${settings.threads}`;
    if (!model || key !== wanted) {
      await model?.dispose(); model = null; key = null;
      send({ type: 'progress', text: 'Loading ProteinMPNN' });
      model = await ProteinMPNN.load(modelManifestURL(settings.family, settings.model), {
        backend: settings.backend, numThreads: settings.threads, wasmPaths, signal,
        onFallback: () => send({ type: 'progress', text: 'ProteinMPNN / WASM' }),
      });
      key = wanted;
    }
    let prepared, lastProgress = 0;
    const prepare = async noiseSeed => {
      send({ type: 'progress', text: 'Preparing ProteinMPNN' });
      return model.prepare(input.structure, { seed: noiseSeed, backboneNoise: settings.backboneNoise, signal });
    };
    const results = [];
    const onProgress = progress => {
      const now = performance.now();
      if (now - lastProgress > 150) {
        send({ type: 'progress', text: `ProteinMPNN ${results.length + 1} / ${count}, ${progress.completed} / ${progress.total}` }); lastProgress = now;
      }
    };
    const save = (result, noiseSeed) => {
      results.push({ sequence: result.sequence, chains: result.chains, seed: result.seed,
        score: result.score, globalScore: result.globalScore, temperature: result.temperature,
        noiseSeed, designMask: result.designMask });
    };
    if (settings.freshNoise && settings.backboneNoise > 0) {
      for (let i = 0; i < count; i++) {
        prepared = await prepare(seed + i);
        save(await prepared.sample({ ...options, temperature: settings.temperature, seed: seed + i, signal, onProgress }), seed + i);
      }
    } else {
      prepared = await prepare(seed);
      for await (const result of prepared.sampleStream({ ...options, temperature: settings.temperature,
        numSequences: count, batchSize: settings.batchSize, seed, signal, onProgress })) save(result, seed);
    }
    send({ type: 'result', results, manifest: model.manifest, provider: model.backend.provider });
  } catch (error) {
    send({ type: 'error', error: signal.aborted ? 'ProteinMPNN cancelled' : error.message, cancelled: signal.aborted });
  } finally { controller = null; }
};
