import pytest
from pathlib import Path

from iiif.exceptions import InvalidIIIFParameter
from iiif.ops import Region, Size, Rotation, Quality, Format, IIIFOps, IIIF_LEVEL
from iiif.profiles.base import ImageInfo

"""
Most of these tests go through the various IIIF Image API v3 compliance levels which can be found
here https://iiif.io/api/image/3.0/compliance/.
"""


@pytest.fixture
def info() -> ImageInfo:
    return ImageInfo('test', 'image1', 4000, 6000)


@pytest.fixture
def full_region(info: ImageInfo) -> Region:
    return Region(0, 0, info.width, info.height, full=True)


class TestParseRegion:

    def test_level0(self, info: ImageInfo):
        region = Region.parse(f'0,0,{info.width},{info.height}', info)
        assert region == Region(0, 0, info.width, info.height, full=True)

    def test_level1_regionByPx(self, info: ImageInfo):
        region = Region.parse('10,10,1000,2400', info)
        assert region == Region(10, 10, 1000, 2400, full=False)

    def test_level1_regionByPx_oob(self, info):
        region = Region.parse('3000,5000,5000,5000', info)
        assert region == Region(3000, 5000, 1000, 1000, full=False)

    def test_level1_regionSquare_portrait(self, info: ImageInfo):
        region = Region.parse('square', info)
        assert region == Region(0, 1000, 4000, 4000, full=False)

    def test_level1_regionSquare_landscape(self):
        info = ImageInfo('test', 'image1', 6000, 4000)
        region = Region.parse('square', info)
        assert region == Region(1000, 0, 4000, 4000, full=False)

    def test_level1_regionSquare_already(self):
        info = ImageInfo('test', 'image1', 4000, 4000)
        region = Region.parse('square', info)
        assert region == Region(0, 0, 4000, 4000, full=True)

    def test_level2_regionByPct(self, info: ImageInfo):
        region = Region.parse('pct:10,10,90,90', info)
        assert region == Region(400, 600, 3600, 5400, full=False)

    def test_level2_regionByPct_oob(self, info):
        region = Region.parse('pct:10,10,100,100', info)
        assert region == Region(400, 600, 3600, 5400, full=False)

    @pytest.mark.parametrize('region', ['', '10,50', '-10,-15', '10,10,0,0'])
    def test_invalid(self, info, region):
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Region.parse(region, info)
        assert exc_info.value.status_code == 400


class TestParseSize:

    def test_level0(self, full_region: Region):
        size = Size.parse('max', full_region)
        assert size == Size(full_region.w, full_region.h, max=True)

    def test_level1_sizeByW(self, full_region: Region):
        size = Size.parse('500,', full_region)
        assert size == Size(500, 750, max=False)

    def test_level1_sizeByH(self, full_region: Region):
        size = Size.parse(',600', full_region)
        assert size == Size(400, 600, max=False)

    def test_level1_sizeByWh(self, full_region: Region):
        size = Size.parse('190,568', full_region)
        assert size == Size(190, 568, max=False)

    def test_level2_sizeByPct(self, full_region: Region):
        size = Size.parse('pct:23.673', full_region)
        assert size == Size(947, 1420, max=False)

    def test_level2_sizeByPct_max(self, full_region: Region):
        size = Size.parse('pct:100', full_region)
        assert size == Size(full_region.w, full_region.h, max=True)

    def test_level2_sizeByPct_max_rounding(self, full_region: Region):
        size = Size.parse('pct:99.999', full_region)
        assert size == Size(full_region.w, full_region.h, max=True)

    def test_level2_sizeByConfinedWh_1(self, full_region: Region):
        size = Size.parse('!568,901', full_region)
        assert size == Size(568, 852, max=False)

    def test_level2_sizeByConfinedWh_2(self, full_region: Region):
        size = Size.parse('!400,600', full_region)
        assert size == Size(400, 600, max=False)

    def test_level2_sizeByConfinedWh_3(self, full_region: Region):
        size = Size.parse('!4000,6000', full_region)
        assert size == Size(4000, 6000, max=True)

    def test_level2_sizeByConfinedWh_4(self, full_region: Region):
        size = Size.parse('!123,5783', full_region)
        assert size == Size(123, 184, max=False)

    def test_level2_sizeByConfinedWh_5(self, full_region: Region):
        size = Size.parse('!225,100', Region(0, 0, 200, 133))
        assert size == Size(150, 100, max=False)

    invalid_scenarios = [
        # blank
        '',
        ',',
        # too many options
        '10,50,40',
        # too few options
        '10',
        # too many options
        '10,10,0,0',
        # upscaling
        '7000,',
        # upscaling
        ',9000',
        # upscaling
        '10000, 12000',
        # upscaling with percentages
        'pct:110',
        # 0s
        'pct:0',
        '0,0',
        '0,100',
        '100,0',
    ]

    @pytest.mark.parametrize('size', invalid_scenarios)
    def test_invalid(self, full_region, size):
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Size.parse(size, full_region)
        assert exc_info.value.status_code == 400


class TestParseRotation:

    def test_level0(self):
        rotation = Rotation.parse('0')
        assert rotation == Rotation(0)

    @pytest.mark.parametrize('angle', [0, 90, 180, 270])
    def test_level2_rotationBy90s(self, angle: int):
        rotation = Rotation.parse(str(angle))
        assert rotation == Rotation(angle)

    @pytest.mark.parametrize('angle', [0, 90, 180, 270])
    def test_level2_mirroring(self, angle: int):
        rotation = Rotation.parse(f'!{angle}')
        assert rotation == Rotation(angle, mirror=True)

    # we don't support arbitrary rotation, nor values outside of 0, 90, 180 and 270
    @pytest.mark.parametrize('angle', [-90, 10, 3, 289, 360, 500, 3600, 'beans'])
    def test_invalid_angles(self, angle: int):
        rotation = str(angle)
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Rotation.parse(rotation)
        assert exc_info.value.status_code == 400

        rotation = f'!{angle}'
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Rotation.parse(rotation)
        assert exc_info.value.status_code == 400


class TestParseQuality:

    def test_level0(self):
        assert Quality.parse('default') == Quality.default

    def test_level2_color(self):
        assert Quality.parse('color') == Quality.color

    def test_level2_colour(self):
        assert Quality.parse('colour') == Quality.color

    def test_level2_gray(self):
        assert Quality.parse('gray') == Quality.gray

    def test_level2_grey(self):
        assert Quality.parse('grey') == Quality.gray

    def test_bitonal(self):
        assert Quality.parse('bitonal') == Quality.bitonal

    def test_invalid(self):
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Quality.parse('banana')
        assert exc_info.value.status_code == 400

    def test_extras(self):
        extras = Quality.extras()
        for quality in Quality:
            for value in quality.value:
                if value == 'default':
                    assert value not in extras
                else:
                    assert value in extras


class TestParseFormat:

    def test_level0(self):
        assert Format.parse('jpg') == Format.jpg

    def test_level2_png(self):
        assert Format.parse('png') == Format.png

    @pytest.mark.parametrize('fmt', ['tif', 'gif', 'pdf', 'jp2', 'webp', 'arms!'])
    def test_other_formats_are_invalid(self, fmt):
        with pytest.raises(InvalidIIIFParameter) as exc_info:
            Format.parse(fmt)
        assert exc_info.value.status_code == 400


class TestParseParams:

    def test_default(self, info: ImageInfo):
        ops = IIIFOps.parse(info)
        assert ops == IIIFOps(
            region=Region(0, 0, info.width, info.height, full=True),
            size=Size(info.width, info.height, max=True),
            rotation=Rotation(0, mirror=False),
            quality=Quality.default,
            format=Format.jpg
        )

    def test_with_options(self, info: ImageInfo):
        ops = IIIFOps.parse(
            info,
            region='10,20,40,50',
            size='10,',
            rotation='!90',
            quality='gray',
            fmt='png'
        )
        assert ops == IIIFOps(
            region=Region(10, 20, 40, 50, full=False),
            size=Size(10, 12, max=False),
            rotation=Rotation(90, mirror=True),
            quality=Quality.gray,
            format=Format.png
        )


def test_ops_location():
    ops = IIIFOps(
        region=Region(10, 20, 40, 50, full=False),
        size=Size(10, 12, max=False),
        rotation=Rotation(90, mirror=True),
        quality=Quality.gray,
        format=Format.png
    )
    assert ops.location == Path('10_20_40_50', '10_12', '-90', 'gray.png')


def test_region_str(full_region):
    assert str(full_region) == 'full'
    assert str(Region(10, 20, 40, 50, full=False)) == '10_20_40_50'


def test_size_str():
    assert str(Size(100, 408, max=False)) == '100_408'
    assert str(Size(100, 408, max=True)) == 'max'


def test_rotation_str():
    assert str(Rotation(180, mirror=False)) == '180'
    assert str(Rotation(180, mirror=True)) == '-180'


def test_level():
    # check we're level 2 baby
    assert IIIF_LEVEL == 2
