from libact.base.interfaces import Labeler


class OurLabeler(Labeler):

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)

    def label(self, feature):
        banner = "Enter the associated label with the article (" + feature + "): "
        lbl = input(banner)

        while (self.label_name is not None) and (lbl not in self.label_name):
            print('Invalid label, please re-enter the associated label.')
            lbl = input(banner)

        return self.label_name.index(lbl)
