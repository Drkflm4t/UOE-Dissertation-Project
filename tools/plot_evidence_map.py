"""Create the compact Chapter 4 RQ-to-evidence map."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("outputs/figures/chapter4_evidence_map.png")
WIDTH, HEIGHT = 2400, 920


def font(size, bold=False):
    name = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def box(draw, bounds, text, fill, outline, *, text_size=30, bold=False, radius=18):
    x0, y0, x1, y1 = bounds
    draw.rounded_rectangle(bounds, radius=radius, fill=fill, outline=outline, width=3)
    fnt = font(text_size, bold=bold)
    spacing = max(8, text_size // 4)
    text_box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    draw.multiline_text(
        ((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - 3),
        text,
        font=fnt,
        fill="#17202a",
        spacing=spacing,
        align="center",
    )


def arrow(draw, x1, x2, y, color):
    draw.line((x1, y, x2 - 12, y), fill=color, width=4)
    draw.polygon([(x2, y), (x2 - 16, y - 10), (x2 - 16, y + 10)], fill=color)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    margin = 38
    gap = 30
    col_widths = [360, 910, 970]
    xs = [margin]
    for width in col_widths[:-1]:
        xs.append(xs[-1] + width + gap)

    header_top, header_bottom = 24, 130
    headers = ["Research question", "Primary evidence", "Supporting evidence"]
    for x, width, title in zip(xs, col_widths, headers):
        box(draw, (x, header_top, x + width, header_bottom), title, "#e9edf2", "#58636f", text_size=31, bold=True)

    rows = [
        (
            "RQ1\nInjection\nsusceptibility",
            "Original vs Manipulated PDFs\nunder Free and Structured",
            "Author-coded Injection reviews\n+ one matched review example",
            "#fce8e6",
            "#c8463a",
        ),
        (
            "RQ2\nLogic-defect\ndiscriminability",
            "Logic change vs Format change\nwithin each setup",
            "Five matched Logic cases\nfor target-defect inspection",
            "#fff1dd",
            "#d37b19",
        ),
        (
            "RQ3\nReview-aspect\ncoverage",
            "Original Free vs Structured\nreview-aspect profiles",
            "Author coding + human-review benchmark\n+ one count-granularity example",
            "#e8f2fb",
            "#2b7db8",
        ),
        (
            "RQ4\nEfficiency and\nrating dispersion",
            "All Counterfactual reviews\nlength, latency, tokens and ratings",
            "Matched human-review\nlength benchmark",
            "#eaf5e5",
            "#4d8b3d",
        ),
    ]

    row_height = 165
    row_gap = 23
    top = 155
    for i, (rq, primary, support, face, edge) in enumerate(rows):
        y0 = top + i * (row_height + row_gap)
        y1 = y0 + row_height
        contents = [rq, primary, support]
        sizes = [29, 30, 28]
        for j, (x, width, text, size) in enumerate(zip(xs, col_widths, contents, sizes)):
            box(
                draw,
                (x, y0, x + width, y1),
                text,
                face if j == 0 else "#ffffff",
                edge,
                text_size=size,
                bold=(j == 0),
            )
        cy = (y0 + y1) // 2
        for j in range(2):
            arrow(draw, xs[j] + col_widths[j] + 5, xs[j + 1] - 5, cy, edge)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, dpi=(240, 240), optimize=True)


if __name__ == "__main__":
    main()
