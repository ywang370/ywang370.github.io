from pathlib import Path
import os
import shutil

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ByteStringObject, ContentStream, NameObject


repo = Path("/Users/yilwang/Desktop/ywang370.github.io")
source = repo / "Yilin_Wang_Resume.pdf"
output_dir = repo / "output" / "pdf"
output = output_dir / "Yilin_Wang_Resume.pdf"
backup = repo / "tmp" / "pdfs" / "resume_source" / "Yilin_Wang_Resume.original.pdf"

output_dir.mkdir(parents=True, exist_ok=True)
backup.parent.mkdir(parents=True, exist_ok=True)
if not backup.exists():
    shutil.copy2(source, backup)

# Always regenerate from the preserved original so the edit is reproducible.
reader = PdfReader(backup)
writer = PdfWriter()
writer.clone_document_from_reader(reader)

# The embedded Carlito Bold font maps these single-byte glyph codes through its
# ToUnicode CMap. Replace only "ICLR 2" with "CVPR 2" in the UniReal entry,
# preserving the original vector text, font, links, and page layout.
old_prefix = b"\x02\x0f\x03\x0b\x05\x19"  # ICLR 2
new_prefix = b"\x0f\x3e\x0a\x0b\x05\x19"  # CVPR 2

page = writer.pages[1]
content = ContentStream(page.get_contents(), writer)
replacement_count = 0
highlight_shift_count = 0

for operands, operator in content.operations:
    if operator == b"Td" and len(operands) == 2:
        x, y = (float(value) for value in operands)
        if abs(x - 447.55) < 0.01 and abs(y - 636.25) < 0.01:
            # "CVPR" is 4.35 pt wider than "ICLR" in the embedded 10 pt font.
            operands[0] = type(operands[0])(451.90)
            highlight_shift_count += 1
    if operator != b"TJ":
        continue
    for index, item in enumerate(operands[0]):
        raw = getattr(item, "original_bytes", None)
        if raw == old_prefix:
            operands[0][index] = ByteStringObject(new_prefix)
            replacement_count += 1

if replacement_count != 1:
    raise RuntimeError(f"Expected one UniReal venue replacement, found {replacement_count}")
if highlight_shift_count != 1:
    raise RuntimeError(f"Expected one Highlight-label shift, found {highlight_shift_count}")

page[NameObject("/Contents")] = writer._add_object(content)

temporary_output = output.with_suffix(".tmp.pdf")
with temporary_output.open("wb") as stream:
    writer.write(stream)

verification_text = "\n".join(
    page.extract_text() or "" for page in PdfReader(temporary_output).pages
)
unireal_start = verification_text.find("UniReal: Universal Image Generation and Editing via Learning Real-World Dynamics")
unireal_entry = verification_text[unireal_start : unireal_start + 180]
if unireal_start < 0 or "CVPR 2025" not in unireal_entry:
    raise RuntimeError("Corrected UniReal venue was not recoverable from the revised PDF")
if "ICLR 2025" in unireal_entry:
    raise RuntimeError("Old UniReal venue remains in the revised PDF")

os.replace(temporary_output, output)
temporary_source = source.with_suffix(".tmp.pdf")
shutil.copy2(output, temporary_source)
os.replace(temporary_source, source)

print(output)
print(source)
