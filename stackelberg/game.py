from .player import Player
import abc

__all__=['Game']

class Game(abc.ABC):
    def __init__(self,Learders:list[Player]=None,Followers:list[Player]=None) -> None:
        if Learders is None:
            self.Leaders=[]
        else:
            self.Leaders=Learders
        if Followers is None:
            self.Followers=[]
        else:
            self.Followers=Followers

        # self.Leaders_opt={} 
        # self.Followers_opt={}
    
    def add_Leader(self,*leaders:Player):
        for i in leaders:
            self.Leaders.append(i)

    def add_Follower(self,*followers:Player):
        for i in followers:
            self.Followers.append(i)

    def setup(self):
        self._check_players_tag()

    def _check_players_tag(self):
        Ltags=[]
        Ftags=[]
        if len(self.Leaders)==0 or len(self.Followers)==0:
            raise ValueError("The Leaders and Follower can't be null")
        for i in self.Leaders:
            itag=i.tag
            if itag in Ltags:
                raise ValueError(f"please give different tags ({itag}) for Players")
            else:
                Ltags.append(itag)
                # self.Leaders_opt[itag]=[]
        for i in self.Followers:
            itag=i.tag
            if itag in Ftags:
                raise ValueError(f"please give different tags ({itag}) for Players")
            else:
                Ftags.append(itag)
                # self.Followers_opt[itag]=[]

    @abc.abstractmethod
    def run_External_driver(self):
        pass
        # for i in self.Followers:
        #     i.run_External_driver(termination=('n_gen',100))
        #     self.Followers_opt[i.tag].append(i.opt)



class NaGame(Game):
    def __init__(self, Learders: list[Player] = None, Followers: list[Player] = None) -> None:
        super().__init__(Learders, Followers)
    
    def run_External_driver(self):
        for i in self.Followers:
            i.run_External_driver(termination=('n_gen',100))
            self.Followers_opt[i.tag]=i.opt
        




