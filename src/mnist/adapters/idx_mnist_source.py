import struct
from mnist.domain.models import Sample, Vector
from typing import Iterator

xtract_format = lambda b: ">"+"I"*(len(b)//4)

def parse_idx_header(source: bytes):
    values = struct.unpack(xtract_format(source), source)
    dims = list(values[1:])
    _, _, data_type, num_dims = struct.unpack('>BBBB', struct.pack('>I', values[0]))
    if not data_type or not num_dims:
        raise ValueError("Magic number must be greater than 0")

    return data_type, num_dims, dims

def parse_idx_images(source: bytes):
    header = source[:16]
    images = source[16:]
    data_type, num_dims, dims = parse_idx_header(header)
    num_images, width, height = dims
    len_image = width * height

    _format = ">"+"B" * len_image

    for i in range(num_images):
        image = images[:len_image]
        image = struct.unpack(_format, image)
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

class IdxMnistSource:
    def __init__(self, images_source, labels_source):
        self.images_source = images_source
        self.labels_source = labels_source

        images_header = images_source[:16]
        labels_header = labels_source[:8]

        _, _, img_dims = parse_idx_header(images_header)
        _, _, lbl_dims = parse_idx_header(labels_header)

        if img_dims[0] != lbl_dims[0]:
            raise ValueError("Images and Labels number are differents")

    def load(self) -> Iterator[Sample]:
        images_generator = parse_idx_images(self.images_source)
        labels_generator = parse_idx_labels(self.labels_source)

        for pixels, label in zip(images_generator, labels_generator):
            pixels = [pixel / 255 for pixel in pixels]
            
            yield Sample(Vector(pixels), Vector.one_hot(label, 10))