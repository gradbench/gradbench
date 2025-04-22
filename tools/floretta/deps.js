const arch = { amd64: "x86_64", arm64: "aarch64" }[process.argv[3]];
const wasmTools = () => {
  const version = "1.226.0";
  return `https://github.com/bytecodealliance/wasm-tools/releases/download/v${version}/wasm-tools-${version}-${arch}-linux.tar.gz`;
};
const floretta = () => {
  const version = "0.5.0";
  return `https://github.com/samestep/floretta/releases/download/v${version}/floretta-${arch}-unknown-linux-musl`;
};
console.log(
  { "wasm-tools": wasmTools(), floretta: floretta() }[process.argv[2]],
);
