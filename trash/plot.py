import matplotlib.pyplot as plt
from dataclasses import dataclass
from noawclg.main import get_noaa_data as gnd


def fmt(data): return data

@dataclass
class plot_data_from_place:
    data:gnd
    place:str
    title:str=f'plot'
    path_file:str='plot_temp.png'
    dpi:int=600
    author:str='by @gpftc_ | @reinanbr_'
    key_noaa:str='tmp80m'
    fmt_data:str=fmt
    max_label:str='max'
    med_label:str='med'
    min_label:str='min'
    xlabel:str='date'
    ylabel:str=''
    show:bool=True
    cla:bool=True
    
    def render(self):
        if self.cla:
            plt.cla()
            plt.clf()
        data_point = self.data.get_data_from_place(self.place)
        temp = self.fmt_data(data_point[self.key_noaa])
        #print('getted data..')
        
        temp = temp.to_pandas()

        self.m_temp = temp.rolling(8).mean()
        self.max_temp = temp.rolling(8).max()
        self.min_temp = temp.rolling(8).min()
        self.index = temp.index

        ax2 = self.max_temp.plot(label=self.max_label,color='red')
        ax1 = self.m_temp.plot(label=self.med_label,color='green')
        ax3 = self.min_temp.plot(label=self.min_label,color='blue')

        plt.title(self.title,fontweight='bold')
        plt.legend()
        #plt.annotate(self.author,xy=(temp.index[10],20))
        plt.text(0.14, 0.05, self.author, fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)
        about_key=self.data.get_noaa_keys()[self.key_noaa]
        key_about = f'{self.key_noaa}:\n {about_key}'
        plt.text(0.65, 0.001, key_about, fontsize=8, transform=plt.gcf().transFigure)
        plt.text(0.65, 0.04, f'data: GFS{self.data.gfs} NOMADS-OpenDAP',color='gray', fontsize=8, transform=plt.gcf().transFigure)
        plt.text(0.19, 0.01, 'NOAA/NASA',color='gray',fontweight='bold', fontsize=9, transform=plt.gcf().transFigure)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        plt.tight_layout()
        #plt.savefig(self.path_file,dpi=self.dpi)
            
        return plt




