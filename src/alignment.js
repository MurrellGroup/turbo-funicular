import bioseq from 'bioseq';
import { AMINO_ACIDS } from './sequence.js';

// Standard BLOSUM62, alphabet ARNDCQEGHILKMFPSTWYV; unknown X scores -1.
const B62 = `4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4`.split('\n').map(row => [...row.split(' ').map(Number), -1]);
B62.push(Array(21).fill(-1));
const alphabet = bioseq.makeAlphabetMap(AMINO_ACIDS + 'X', 20);

export function alignChain(chain, query) {
  if ((chain.sequence.length + 1) * (query.length + 1) > 8000000) throw new Error(`Chain ${chain.id} exceeds the alignment memory limit.`);
  const result = bioseq.align(chain.sequence, query, true, B62, [9.5, 0.5], undefined, alphabet) ?? { score: 0 };
  const columns = [];
  let i = 0, j = 0, aligned = 0, matches = 0;
  const add = (reference, input, mapped) => columns.push({ reference, input, mapped,
    residue: reference === '-' ? null : chain.residues[i++], queryIndex: input === '-' ? null : j++ });
  if (result.score > 0 && result.CIGAR?.length) {
    while (i < result.position) add(chain.sequence[i], '-', false);
    for (const op of result.CIGAR) {
      const type = op & 15, length = op >>> 4;
      for (let k = 0; k < length; k++) {
        if (type === 0) {
          matches += chain.sequence[i] === query[j] ? 1 : 0; aligned++;
          add(chain.sequence[i], query[j], true);
        } else if (type === 2) add(chain.sequence[i], '-', false);
        else if (type === 1 || type === 4) add('-', query[j], false);
        else throw new Error('Unsupported alignment operation.');
      }
    }
  }
  while (i < chain.sequence.length) add(chain.sequence[i], '-', false);
  while (j < query.length) add('-', query[j], false);
  return { id: chain.id, columns, score: result.score || 0, aligned, matches,
    identity: aligned ? matches / aligned : 0, coverage: aligned / chain.sequence.length };
}
