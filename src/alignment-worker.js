import { alignChain } from './alignment.js';
self.onmessage = ({ data }) => {
  try { self.postMessage({ results: data.chains.map(chain => alignChain(chain, data.sequence)) }); }
  catch (error) { self.postMessage({ error: error.message }); }
};
