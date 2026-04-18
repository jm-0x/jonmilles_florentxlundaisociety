export function displayToken(tok: string): string {
  if (tok === "") return "\u00B7";
  if (tok.startsWith(" ")) return "\u00B7" + tok.slice(1);
  return tok;
}

export function layerLabel(idx: number): string {
  return idx === 0 ? "EMB" : `L${idx - 1}`;
}
