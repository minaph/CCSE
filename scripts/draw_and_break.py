from PIL import Image, ImageDraw, ImageFont
import subprocess

def get_fonts():
    # fc-listコマンドを実行
    result = subprocess.run(["fc-list.exe"], stdout=subprocess.PIPE)

    # 実行結果を取得
    # C:/texlive/2021/texmf-dist/fonts/truetype/google/noto/NotoSans-ExtraBoldItalic.ttf: Noto Sans,Noto Sans ExtraBold:style=ExtraBold Italic,Italic
    output = result.stdout.decode("utf-8")


    fonts = []
    # フォントの一覧を出力
    for font in output.split("\n"):
        _font = dict()
        _font["source"] = font
        font_data = [x.strip() for x in font.split(":")]
        if font_data[0] == "C":
            _font["path"] = "C:" + font_data[1]
            index = 2
        else:
            _font["path"] = font_data[0]
            index = 1
        if len(font_data) > index:
            _font["names"] = font_data[index].split(",")
            if len(font_data) > index + 1:
                _font["styles"] = font_data[index + 1][len("style=") :].split(",")
            else:
                _font["styles"] = []
                # print("irregular font: NO STYLE")
        else:
            _font["names"] = []
            # print("irregular font: NO NAME")

        # _font["styles"] = font_data[index + 1][len("style=") :].split(",")
        # print(_font)
        fonts.append(_font)

    return fonts

target_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# # フォントの一覧を取得

# 描画する文字
text = "松"

def draw(text, font_path):

    # フォントの指定 (デフォルトのものを使用)
    # font = ImageFont.load_default()
    font = ImageFont.truetype(font_path, 100)

    # print(font)

    # 画像を作成 (120x120ピクセル、白で塗りつぶす)
    image = Image.new("RGB", (120, 120), (255, 255, 255))

    # 描画するための画像を取得
    draw = ImageDraw.Draw(image)

    # 文字を描画
    draw.text((10, -20), text, font=font, fill=(0, 0, 0))

    # 画像を保存
    # image.save("text.png")
    return image

# draw(text, target_font).show()



# フォントでサポートされている文字の一覧を取得



from fontTools import ttLib

def get_all_chars(font_file):
    cmaps = []
    for font_number in range(4):
        with ttLib.ttFont.TTFont(font_file, fontNumber=font_number) as font:
            # glyph_set = font.getGlyphSet()  # {グリフ名: グリフ} っぽいオブジェクト
            cmap = font.getBestCmap()       # {Unicode: グリフ名}
            cmaps = cmaps + list(cmap.keys())
    return list(cmap.keys())


# print([chr(x) for x in get_all_chars(target_font)])
# for char in get_all_chars(target_font):
    # draw(chr(char), target_font).save(f"scripts/drawings/{char}.png")