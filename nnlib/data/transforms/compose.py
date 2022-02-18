from nnlib.data.transforms.transform import BaseTransform


class IndexCompose(BaseTransform):

    def __init__(self, transforms):
        super(IndexCompose, self).__init__()
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self.transforms) + idx
        if not (0 <= idx < len(self.transforms)):
            raise IndexError(f"[ERROR:DATA] Composed transforms have {len(self.transforms)} elements, "
                             f"but indexing {idx}.")
        return self.transforms[idx]

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
