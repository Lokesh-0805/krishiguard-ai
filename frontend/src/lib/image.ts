// Shrink large photos in the browser before upload. The model only sees a 224 px crop, so a 1600 px image loses
// nothing for diagnosis, while keeping requests well under hosting proxy limits (~4.5 MB on Vercel) and fast on
// mobile data. Small images are sent unchanged.
export async function prepareForUpload(file: File, maxSide = 1600, maxBytes = 1.5 * 1024 * 1024): Promise<File> {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size <= maxBytes) return file;
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * scale);
    c.height = Math.round(bmp.height * scale);
    c.getContext("2d")?.drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise<Blob | null>((r) => c.toBlob(r, "image/jpeg", 0.92));
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch {
    return file; // decoding unsupported in this browser: let the server validate the original
  }
}
