import struct
from mnist.domain.models import NetSample, Vector
from mnist.ports.sample_source import SampleSource
from typing import Iterator

xtract_format = lambda b: ">"+"I"*(len(b)//4)

UBYTE = 0x08  # unico tipo que leen los parsers (struct 'B'), y el que usa MNIST

def parse_idx_header(source: bytes):
    if not source or len(source) % 4:
        raise ValueError(
            f"IDX header length must be a positive multiple of 4 bytes, got {len(source)}: "
            "empty or truncated file"
        )
    values = struct.unpack(xtract_format(source), source)
    dims = list(values[1:])
    zero_1, zero_2, data_type, num_dims = struct.unpack('>BBBB', struct.pack('>I', values[0]))
    if zero_1 or zero_2:
        raise ValueError(
            f"IDX magic number must start with two zero bytes, got {zero_1:#04x} {zero_2:#04x}"
        )
    if data_type != UBYTE:
        raise ValueError(
            f"Unsupported IDX data type {data_type:#04x}: only unsigned byte ({UBYTE:#04x}) is supported"
        )
    if num_dims != len(dims):
        raise ValueError(
            f"IDX header declares {num_dims} dimension(s) but {len(dims)} size field(s) were read: "
            "wrong kind of file (images vs labels) or truncated header"
        )

    return data_type, num_dims, dims

def parse_idx_images(source: bytes):
    header = source[:16]
    images = source[16:]
    data_type, num_dims, dims = parse_idx_header(header)
    num_images, width, height = dims
    len_image = width * height

    _format = ">"+"B" * len_image

    for i in range(num_images):
        raw_image = images[:len_image]
        image = struct.unpack(_format, raw_image)
        images = images[len_image:]
        yield image

def parse_idx_labels(source: bytes):
    header = source[:8]
    data_type, num_dims, dims = parse_idx_header(header)
    num_labels = dims[0]
    labels = source[8:]
    for i in range(num_labels):
        label = labels[i]
        yield label        

class IdxMnistSource(SampleSource):
    def __init__(self, images_source, labels_source):
        self.images_source = images_source
        self.labels_source = labels_source

        images_header = images_source[:16]
        labels_header = labels_source[:8]

        _, _, img_dims = parse_idx_header(images_header)
        _, _, lbl_dims = parse_idx_header(labels_header)

        if img_dims[0] != lbl_dims[0]:
            raise ValueError("Images and Labels number are differents")

        num_images, width, height = img_dims
        expected_images = 16 + num_images * width * height
        if len(images_source) != expected_images:
            raise ValueError(
                f"IDX images file has {len(images_source)} bytes but its header "
                f"({num_images} images of {width}x{height}) implies {expected_images}: "
                "truncated or corrupt file"
            )

        expected_labels = 8 + lbl_dims[0]
        if len(labels_source) != expected_labels:
            raise ValueError(
                f"IDX labels file has {len(labels_source)} bytes but its header "
                f"({lbl_dims[0]} labels) implies {expected_labels}: truncated or corrupt file"
            )

    def load(self) -> Iterator[NetSample]:
        images_generator = parse_idx_images(self.images_source)
        labels_generator = parse_idx_labels(self.labels_source)

        for pixels, label in zip(images_generator, labels_generator):
            pixels = [pixel / 255 for pixel in pixels]
            
            yield NetSample(Vector(pixels), Vector.one_hot(label, 10))