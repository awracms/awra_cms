from collections import OrderedDict

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from awrams.visualisation import utils

A4 = [8.267,11.692]

class PageLayout:
    def __init__(self,size=A4,child=None,fonts=None,**kwargs):
        if child is not None:
            self.child = child
            self.child._set_parent(self)
        self.size = size
        self.bounds = dict(left=0.05,right=0.95,top=0.98,bottom=0.05,wspace=0.0,hspace=0.0)
        self.bounds.update(**kwargs)
        self.fonts = fonts

    def plot(self):
        self._figure = plt.figure(figsize=self.size)
        gs = GridSpec(1,1)
        gs.update(**self.bounds)
        self.child.plot(gs[0,0])

    def set_child(self,child):
        self.child = child
        child._set_parent(self)

    def _get_figure(self):
        return self._figure

    def _get_font(self,key):
        return self.fonts[key]

class View:
    def _set_parent(self,parent):
        self.parent = parent

    def _get_figure(self):
        return self.parent._get_figure()

    def _get_font(self,key):
        return self.parent._get_font(key)

class ContainerView(View):
    def __init__(self):
        self.names = {}

    def __setitem__(self,key,value):
        if key in self.names:
            key = self.names[key]

        if not key in range(self.nchildren):
            raise Exception('Invalid item number %s', key)

        self.children[key] = value
        value._set_parent(self)

    def __getitem__(self,key):
        if key in self.names:
            return self.children[self.names[key]]
        else:
            return self.children[key]

    def set_name(self,name,item):
        if not item in range(self.nchildren):
            raise Exception('Invalid item number %s', item)
        self.names[name] = item

class VerticalSplit(ContainerView):
    def __init__(self,ratios,wspace=0.0,hspace=0.0):
        ContainerView.__init__(self)
        self.ratios = ratios
        self.spacing = dict(wspace=wspace,hspace=hspace)
        self.children = {}
        self.nchildren = len(self.ratios)

    def plot(self,subspec):
        gs = GridSpecFromSubplotSpec(self.nchildren, 1, height_ratios=self.ratios,subplot_spec=subspec,**self.spacing)
        for row in range(self.nchildren):
            child = self.children.get(row)
            if child is not None:
                child.plot(gs[row,0])

class HorizontalSplit(ContainerView):
    def __init__(self,ratios,wspace=0.0,hspace=0.0):
        ContainerView.__init__(self)
        self.ratios = ratios
        self.spacing = dict(wspace=wspace,hspace=hspace)
        self.children = {}
        self.nchildren = len(self.ratios)

    def plot(self,subspec):
        gs = GridSpecFromSubplotSpec(1, self.nchildren, width_ratios=self.ratios,subplot_spec=subspec,**self.spacing)
        for col in range(self.nchildren):
            child = self.children.get(col)
            if child is not None:
                child.plot(gs[0,col])


class PaddedView(View):
    def __init__(self,child,padding=None,**kwargs):
        pad_actual = dict(lpad=0.,rpad=0.,tpad=0.,bpad=0.)
        if padding is not None:
            pad_actual.update(padding)
        else:
            pad_actual.update(**kwargs)
        self.padding = pad_actual
        self.child = child
        self.child._set_parent(self)

    def plot(self,subspec):
        fig = self._get_figure()
        size = fig.get_size_inches()
        bounds = bounds_dict(subspec.get_position(fig).bounds)

        new_bounds = {}

        new_bounds['left'] = bounds['left'] + self.padding['lpad']/size[0]
        new_bounds['right']  = bounds['right'] - self.padding['rpad']/size[0]
        new_bounds['top']  = bounds['top'] - self.padding['tpad']/size[1]
        new_bounds['bottom']  = bounds['bottom'] + self.padding['bpad']/size[1]

        gs = GridSpec(1,1,wspace=0.0,hspace=0.0)
        gs.update(**new_bounds)

        self.child.plot(gs[0,0])

class DummyPlotView(View):
    def plot(self,subspec):
        ax = plt.subplot(subspec)
        ax.axis('off')

class WrappedPlotView(View):
    def __init__(self,plot_func,plot_args):
        self.plot_func = plot_func
        self.plot_args = plot_args

    def plot(self,subspec):
        ax = plt.subplot(subspec)
        self.plot_func(ax,**self.plot_args)

class TextView(View):
    def __init__(self,text,x=0.5,y=0.5,valign='center',halign='center',font=None,**kwargs):
        self.text = text
        if font is None:
            font = {}
        self.font = font
        self.valign = valign
        self.halign = halign
        self.kwargs = kwargs
        self.x = x
        self.y = y

    def plot(self,subspec):
        ax = plt.subplot(subspec)
        ax.axis('off')

        font = self._get_font(self.font)

        plt.text(self.x,self.y,self.text,fontdict=font,verticalalignment=self.valign,horizontalalignment=self.halign,**self.kwargs)

class TableView(View):
    def __init__(self,data,title=None,fontsize=7.,titlefont=None,scale=(1.0,1.0)):
        self.data = data
        self.scale = scale
        self.title = title
        self.fontsize = 7.
        if titlefont is None:
            self.titlefont = {'size': '10', 'weight': 'bold'}

    def plot(self,subspec):
        ax = plt.subplot(subspec)

        table = utils.plot_array_table(self.data,ax,scale=self.scale)

        table.auto_set_font_size(False)
        table.set_fontsize(self.fontsize)

        rule = utils.MULTI_COL if self.data.shape[1] > 2 else utils.SERIES

        utils.align_table(table,self.data.shape,rule)
        utils.scale_width(table,self.data.shape)        

        scale_fac = 9.0

        table.scale(self.scale[0],self.fontsize/scale_fac * self.scale[1])

        if self.title is not None:
            ax.set_title(self.title,fontdict=self.titlefont)


def bounds_dict(bounds):
    return dict(
        left = bounds[0],
        right = bounds[0] + bounds[2],
        bottom = bounds[1],
        top = bounds[1] + bounds[3],
        width = bounds[2],
        height = bounds[3]
    )

def pad(view,**kwargs):
    return PaddedView(view,**kwargs)