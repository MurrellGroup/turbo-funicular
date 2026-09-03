function atomLine(record, serial, name, residue, chain, residueNumber, x, y, z, element) {
  return `${record.padEnd(6)}${String(serial).padStart(5)} ${name.padStart(4)} ${residue.padStart(3)} ${chain}${String(residueNumber).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00 20.00          ${element.padStart(2)}`;
}

function linkLine(first, second) {
  const line = Array(80).fill(" ");
  const write = (offset, width, value, alignRight = false) => {
    const text = String(value)[alignRight ? "padStart" : "padEnd"](width).slice(0, width);
    for (let index = 0; index < width; index += 1) line[offset + index] = text[index];
  };
  write(0, 6, "LINK");
  write(12, 4, first.atom, true);
  write(17, 3, first.residue, true);
  write(21, 1, first.chain);
  write(22, 4, first.number, true);
  write(42, 4, second.atom, true);
  write(47, 3, second.residue, true);
  write(51, 1, second.chain);
  write(52, 4, second.number, true);
  return line.join("").trimEnd();
}

export function miniPdb({ component = "BEN", includeLink = false } = {}) {
  const lines = [
    "TITLE     BROWSER PREPARATION FIXTURE",
    atomLine("ATOM", 1, "N", "ALA", "A", 1, -3.0, 0.0, 0.0, "N"),
    atomLine("ATOM", 2, "CA", "ALA", "A", 1, -2.0, 0.0, 0.0, "C"),
    atomLine("ATOM", 3, "C", "ALA", "A", 1, -1.0, 0.0, 0.0, "C"),
    atomLine("ATOM", 4, "O", "ALA", "A", 1, -0.3, 0.8, 0.0, "O"),
    atomLine("ATOM", 5, "CB", "ALA", "A", 1, -2.0, 1.5, 0.0, "C"),
    atomLine("ATOM", 6, "N", "GLY", "A", 2, 0.3, -0.7, 0.0, "N"),
    atomLine("ATOM", 7, "CA", "GLY", "A", 2, 1.5, -0.4, 0.0, "C"),
    atomLine("ATOM", 8, "C", "GLY", "A", 2, 2.4, 0.4, 0.0, "C"),
    atomLine("ATOM", 9, "O", "GLY", "A", 2, 3.5, 0.1, 0.0, "O"),
  ];
  const radius = 1.4;
  for (let index = 0; index < 6; index += 1) {
    const angle = index * Math.PI / 3;
    lines.push(atomLine(
      "HETATM", 10 + index, `C${index + 1}`, component, "B", 101,
      radius * Math.cos(angle), radius * Math.sin(angle), 2.0, "C",
    ));
  }
  if (includeLink) {
    lines.push(linkLine(
      { atom: "CB", residue: "ALA", chain: "A", number: 1 },
      { atom: "C1", residue: component, chain: "B", number: 101 },
    ));
  }
  lines.push(
    "CONECT   10   11   11   15",
    "CONECT   11   10   10   12",
    "CONECT   12   11   13   13",
    "CONECT   13   12   12   14",
    "CONECT   14   13   15   15",
    "CONECT   15   14   14   10",
    "END",
  );
  return `${lines.join("\n")}\n`;
}

export function miniCcd() {
  return `data_BEN
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
BEN C1 C
BEN C2 C
BEN C3 C
BEN C4 C
BEN C5 C
BEN C6 C
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
BEN C1 C2 DOUB Y
BEN C2 C3 SING Y
BEN C3 C4 DOUB Y
BEN C4 C5 SING Y
BEN C5 C6 DOUB Y
BEN C6 C1 SING Y
#
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
BEN SMILES_CANONICAL RDKit 2026.03.5 c1ccccc1
#
`;
}
