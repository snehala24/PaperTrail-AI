import "@testing-library/jest-dom";

// Polyfill Blob.prototype.arrayBuffer if not present in jsdom
if (typeof Blob !== "undefined" && !Blob.prototype.arrayBuffer) {
  Blob.prototype.arrayBuffer = function (): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(reader.result);
        } else {
          // If result is string or something else, convert
          const enc = new TextEncoder();
          resolve(enc.encode(String(reader.result)).buffer);
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(this);
    });
  };
}
