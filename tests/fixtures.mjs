function atomLine(record, serial, name, residue, chain, residueNumber, x, y, z, element) {
  return `${record.padEnd(6)}${String(serial).padStart(5)} ${name.padStart(4)} ${residue.padStart(3)} ${chain}${String(residueNumber).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00 20.00          ${element.padStart(2)}`;
}

export function miniPdb() {
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
      "HETATM", 10 + index, `C${index + 1}`, "BEN", "B", 101,
      radius * Math.cos(angle), radius * Math.sin(angle), 2.0, "C",
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
