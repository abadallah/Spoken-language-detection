from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PLot_CR import PLot_CR

class Error_anaylsis():
    def __init__(self, y_true, y_pred, classes, PathToSavePLots = "", Df_Path="", file_names= None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.save_path = PathToSavePLots
        self.Df_Path=Df_Path
        self.file_names=file_names
        self.caculted_TF_DF=False
        self.TF_DF= None
    
    def CM(self, plottitle, saveing=True):
        result = confusion_matrix(self.y_true,  self.y_pred)
        df_cm = pd.DataFrame(result, index = [i for i in self.classes],
                      columns = [i for i in self.classes])
        plt.figure(figsize = (10,7))
        plt.title(plottitle)
        sns.heatmap(df_cm, annot=True, fmt='g')
        
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plottitle}.jpg")
        plt.show()

        
    def CR(self, printing=True, ploting=False, plotTitle="CR", Saving=False ):
        cr = classification_report(self.y_true,  self.y_pred, target_names=self.classes)
        if printing == True:
            print(cr)
            
        if ploting == True:
            PLot_CR().plot_classification_report(cr, self.classes, 7, title=plotTitle, save = True, save_path=self.save_path )
   
    
    def Create_TF_DF(self):
        if(self.file_names)==None:
            raise Exception("you shouid enter files names in the order of the predictions")
        
        True_OF_False = (self.y_pred == self.y_true).astype(int).tolist()
        self.TF_DF = pd.DataFrame({"filename" : self.file_names, "T_F" : True_OF_False})
        
        Originaldf = pd.read_csv(self.Df_Path)
        Originaldf =  Originaldf[Originaldf["filename"].isin(self.TF_DF["filename"].values)]
        
        self.TF_DF = pd.merge( Originaldf, self.TF_DF, on=["filename"])
        self.caculted_TF_DF =True
        
    
    def All_Length_disPlot(self, PLotTorF=1, plottitle="Length_disPlot", saveing=False, plotType="kde"):
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
            
        sns.displot(data = self.TF_DF[self.TF_DF.T_F==PLotTorF], x="length", kind=plotType, hue="language" ).set(title=plottitle)
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plottitle}.jpg")
            plt.show()  
            
    def Individually_Length_disPlot(self, PLotTorF=1, plottitle="Length_disPlot", saveing=False, plotType="kde", langs="All", Train=False):
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
            
        if langs =="All":
            langs= [lang for lang in self.classes]
            
        if PLotTorF==1:
            Legend_Name= "True Predicted"
        else:
            Legend_Name= "False Predicted"
            
        if Train==True:
            Traindf = pd.read_csv(self.Df_Path)
            
            
        for lang in langs:
            dataPredicted = self.TF_DF[(self.TF_DF.T_F==PLotTorF) & (self.TF_DF.language==lang)]
            dataTest = self.TF_DF[self.TF_DF.language==lang]
            dataTrain = Traindf[Traindf.language==lang]

            if Train==True:
                ax = dataTrain.length.plot(kind = 'density',  alpha = 0.9, color="b", label = "Train", linewidth=2)
                
            ax = dataTest.length.plot(kind = 'density',  alpha = 0.4,color="g", label = 'Test', linewidth=4)
            ax = dataPredicted.length.plot(kind = 'density',  alpha = 0.7, color="r", label = Legend_Name, linewidth=2)

            
            ax.set(xlabel="Length", ylabel="density")
            ax.set(title=f"{plottitle}_{lang}")
            ax.legend(fontsize=8)
            if saveing ==True:
                plt.savefig(f"{self.save_path}/{plottitle}_{lang}.jpg")
            plt.show()
            
            
    def All_Speakers_disPlot(self, PLotTorF=1, plottitle="Speakers_disPlot", saveing=False, Train=False, TopK=5):
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
        if PLotTorF==1:
            Legend_Name= "True Predicted"
        else:
            Legend_Name= "False Predicted"
            
        All_Test = self.TF_DF.groupby("speaker")["T_F"].agg("count")
        Predicted = self.TF_DF[self.TF_DF["T_F"]==PLotTorF].groupby("speaker")["T_F"].agg("count").sort_values(ascending=False)[0:TopK]   
        Renamed_Speakrs = [i+1 for i in range(len(Predicted))]
        
        if Train==False:
            # ax = sns.barplot(x =Renamed_Speakrs , y=TrainSPCount[Falser.index].values,color='b', alpha=0.8 )
            ax = sns.barplot(x =Renamed_Speakrs , y=Predicted[Predicted.index].values ,color='g', alpha=0.8 )
            ax = sns.barplot(x =Renamed_Speakrs , y=Predicted.values ,color='r', alpha=0.8 )

            plt.legend(title='Speaker count', labels=['Test', Legend_Name])

            ax = plt.gca()
            leg = ax.get_legend()
            leg.legendHandles[0].set_color('blue')
            leg.legendHandles[1].set_color('green')
            leg.legendHandles[2].set_color('red')

            ax.set(xlabel="Speaker", ylabel="Count")
            ax.set(title=plottitle)
        else:
            Traindf = pd.read_csv(self.Df_Path)
            TrainSPCount = Traindf.groupby("speaker")["speaker"].agg("count")
            
            ax = sns.barplot(x =Renamed_Speakrs , y=TrainSPCount[Predicted.index].values,color='b', alpha=0.8 )
            ax = sns.barplot(x =Renamed_Speakrs , y=Predicted[Predicted.index].values ,color='g', alpha=0.8 )
            ax = sns.barplot(x =Renamed_Speakrs , y=Predicted.values ,color='r', alpha=0.8 )

            plt.legend(title='Speaker count', labels=['Train', 'Test', Legend_Name])

            ax = plt.gca()
            leg = ax.get_legend()
            leg.legendHandles[0].set_color('blue')
            leg.legendHandles[1].set_color('green')
            leg.legendHandles[2].set_color('red')

            ax.set(xlabel="Speaker", ylabel="Count")
            ax.set(title=plottitle)
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plottitle}.jpg")
        plt.show()