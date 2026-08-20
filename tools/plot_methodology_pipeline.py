"""Create the six-stage Chapter 3 methodology pipeline."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("outputs/figures/chapter3_methodology_pipeline.png")
WIDTH, HEIGHT = 2400, 1160


def font(size: int, bold: bool = False):
    filename = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def centred_text(
    draw,
    bounds,
    text,
    *,
    size,
    fill="#17202a",
    bold=False,
    spacing=11,
):
    x0, y0, x1, y1 = bounds
    fnt = font(size, bold=bold)
    text_bounds = draw.multiline_textbbox(
        (0, 0), text, font=fnt, spacing=spacing, align="center"
    )
    width = text_bounds[2] - text_bounds[0]
    height = text_bounds[3] - text_bounds[1]
    draw.multiline_text(
        ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - 3),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def stage(draw, bounds, number, title, body, fill, outline):
    x0, y0, x1, y1 = bounds
    header_bottom = y0 + 112

    draw.rounded_rectangle(
        bounds,
        radius=25,
        fill="#ffffff",
        outline=outline,
        width=4,
    )
    draw.rounded_rectangle(
        (x0, y0, x1, header_bottom),
        radius=25,
        fill=fill,
        outline=outline,
        width=4,
    )
    draw.rectangle(
        (x0 + 2, header_bottom - 28, x1 - 2, header_bottom),
        fill=fill,
    )

    centred_text(
        draw,
        (x0 + 18, y0 + 10, x1 - 18, y0 + 44),
        f"STAGE {number}",
        size=23,
        fill=outline,
        bold=True,
    )
    centred_text(
        draw,
        (x0 + 18, y0 + 45, x1 - 18, header_bottom - 8),
        title,
        size=36,
        bold=True,
    )
    centred_text(
        draw,
        (x0 + 30, header_bottom + 15, x1 - 30, y1 - 18),
        body,
        size=30,
        spacing=15,
    )


def right_arrow(draw, x1, x2, y):
    colour = "#68737e"
    draw.line((x1, y, x2 - 18, y), fill=colour, width=5)
    draw.polygon(
        [(x2, y), (x2 - 23, y - 14), (x2 - 23, y + 14)],
        fill=colour,
    )


def left_arrow(draw, x1, x2, y):
    colour = "#68737e"
    draw.line((x1, y, x2 + 18, y), fill=colour, width=5)
    draw.polygon(
        [(x2, y), (x2 + 23, y - 14), (x2 + 23, y + 14)],
        fill=colour,
    )


def down_arrow(draw, x, y1, y2):
    colour = "#68737e"
    draw.line((x, y1, x, y2 - 18), fill=colour, width=5)
    draw.polygon(
        [(x, y2), (x - 14, y2 - 23), (x + 14, y2 - 23)],
        fill=colour,
    )


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    margin_x = 65
    gap_x = 75
    box_width = 705
    box_height = 425
    top_y = 65
    bottom_y = 670
    xs = [margin_x + i * (box_width + gap_x) for i in range(3)]

    stages = [
        (
            1,
            "Data preparation",
            "Research papers\n\nCounterfactual dataset\nOriginal full texts and PDFs",
            "#eef1f4",
            "#59636d",
        ),
        (
            2,
            "Controlled inputs",
            "Counterfactual track\nOriginal / Logic / Format\n\nInjection track\nOriginal PDF / Manipulated PDF",
            "#fff0de",
            "#b86a13",
        ),
        (
            3,
            "Review generation",
            "Same reviewer model\n\nFree prose\nStructured schema",
            "#e7f1fb",
            "#2878ad",
        ),
        (
            4,
            "Outcome collection",
            "Free: independent Judge\nStructured: schema fields\n\nShared derived measures",
            "#e9f5e5",
            "#4f863c",
        ),
        (
            5,
            "Evidence support",
            "Condition-masked author coding\nPublic human-review benchmark\nTargeted case audits",
            "#f0eafb",
            "#7050a0",
        ),
        (
            6,
            "Analysis",
            "Matched Free--Structured comparisons\n\nEvidence integration for RQ1--RQ4",
            "#f3ece7",
            "#8a5c3b",
        ),
    ]

    top_bounds = [
        (x, top_y, x + box_width, top_y + box_height)
        for x in xs
    ]
    # The second row runs right-to-left to preserve one continuous reading path.
    bottom_bounds = [
        (xs[2], bottom_y, xs[2] + box_width, bottom_y + box_height),
        (xs[1], bottom_y, xs[1] + box_width, bottom_y + box_height),
        (xs[0], bottom_y, xs[0] + box_width, bottom_y + box_height),
    ]

    for bounds, data in zip(top_bounds + bottom_bounds, stages):
        stage(draw, bounds, *data)

    top_mid_y = top_y + box_height // 2
    bottom_mid_y = bottom_y + box_height // 2

    right_arrow(draw, top_bounds[0][2] + 8, top_bounds[1][0] - 8, top_mid_y)
    right_arrow(draw, top_bounds[1][2] + 8, top_bounds[2][0] - 8, top_mid_y)
    down_arrow(
        draw,
        (top_bounds[2][0] + top_bounds[2][2]) // 2,
        top_bounds[2][3] + 8,
        bottom_bounds[0][1] - 8,
    )
    left_arrow(draw, bottom_bounds[0][0] - 8, bottom_bounds[1][2] + 8, bottom_mid_y)
    left_arrow(draw, bottom_bounds[1][0] - 8, bottom_bounds[2][2] + 8, bottom_mid_y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, dpi=(240, 240), optimize=True)


if __name__ == "__main__":
    main()
