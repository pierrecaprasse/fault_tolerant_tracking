import os
import matplotlib.pyplot as plt
from createGraph import createGraph

#tc = 'PV_BAT_GAS'
#tc = 'PV_HP_BAT'
#tc = 'PV_HP_BAT_WITH_SUPPORT'
#Evaltype = 'det'
#case = 'res_1_newstart'
#case = 'res_1'
#case = 'res_5nov'
#case = 'res_12nov'
#case = 'res_14nov'
#last_gen = True
#gen = [10] 
#gen = [1,2,3,10,50,130]

def plot(tc,Evaltype,case,last_gen):

    path = os.path.dirname(os.path.abspath(__file__))
    pareto_file = os.path.join(os.path.abspath(os.path.join(path, os.pardir)),
                                'RESULTS',
                                tc,
                                Evaltype,
                                case,
                                'pareto',
                                )
    
    paretoSolutions_file = os.path.join(os.path.abspath(os.path.join(path, os.pardir)),
                                        'RESULTS',
                                        tc,
                                        Evaltype,
                                        case,
                                        'paretoSolutions',
                                        )
                                 
    with open(pareto_file,'r') as f:
        lines = f.readlines()
        count_split = 0
        for string in lines:
            if '-' in string:
                count_split += 1
        n_gen = count_split
        n_pop = int(len(lines)/count_split-1)
        print('max gen: %i' % n_gen)
    
    with open(paretoSolutions_file,'r') as f:
        xlines = f.readlines()
    
    if last_gen:
        Y = []
        for l in lines[-n_pop-1:-1]:
            Y.append([float(i) for i in l.split()])
        Yres = list(map(list, zip(*Y)))
        y0 = sorted(Yres[0])
        y1 = [x for _,x in sorted(zip(Yres[0],Yres[1]))]
        #plt.plot(y0,y1,'-o')
        #plt.show()
    
        X = []
        for l in xlines[-n_pop-1:-1]:
            X.append([float(i) for i in l.split()])
        Xres = list(map(list, zip(*X)))
        X_res = []
        for i in range(len(Xres)):
            X_res.append([x for _,x in sorted(zip(Yres[0],Xres[i]))])
            #plt.plot(y0,X_res[i],'-o')
        #plt.show()
    
        #xlabel = "SSR"
        #xunit = "%"
        #ylabel = "LCOE"
        #yunit = "€/MWh"
        
        #grey = [ 0.8-(i/len(gen))*0.8 for i in range(len(gen)-1) ]
        #str1 = [str(i) for i in grey]
        
        #yref = [[y1]]
        #xref = y0
        #X = [y0]
        #Y = [y1]
        #createGraph(X,Y,['-o'] *len(X), ['']*len(X), [X[-1][0]]*len(X),[Y[-1][0]]*len(X), xlabel,xunit,ylabel,yunit, 0.75, -0.18, -0.27, 0.55, [round(min(X[-1]),0),round(max(X[-1]),0),min(Y[-1]),max(Y[-1])],path + r'\\res','svg', addTickY=1,extraTickY=[Y[0][12]],addTickX=1,extraTickX=[X[0][12]],extraColor=False, ColorNames =['0'], font = 'sans-serif')
    
    
    
        with open(os.path.join(path,tc+'_'+Evaltype+'_'+case), "w+") as f:
        
            f.write("generation: " + str(n_gen) +"\n")
            for i in range(len(Y)):
                f.write("sample " + str(i+1)+"\n")
                f.write("input:"+"\n")
                for j in range(len(X_res)):
                    f.write('%f ' %(X_res[j][i]))
                f.write('\n')
                f.write("output:"+"\n")
                f.write('%f %f \n' %(y0[i],y1[i]))
                
            for i in range(len(Y)):
                f.write('[')
                for j in range(len(X_res)):
                    f.write('%f, ' %(X_res[j][i]))
                f.write('],')
                f.write('\n')
                
    
    
    else:
        y0 = []
        y1 = []
        for g in gen:
            Ytemp = []
            for l in lines[(n_pop+1)*g:(n_pop+1)*(g+1)-1]:
                Ytemp.append([float(i) for i in l.split()])
            Yres = list(map(list, zip(*Ytemp)))
            y0.append(sorted(Yres[0]))
            y1.append([x for _,x in sorted(zip(Yres[0],Yres[1]))])
        #for i in range(len(gen)):
            #plt.plot(y1[i],y0[i],'-', )
        #plt.show()            


    return y0,y1,X_res



tclist = ['PV_BAT_GAS','PV_HP_BAT','PV_HP_BAT_WITH_SUPPORT']
#tclist = ['PV_BAT_GAS']
#tclist = ['PV_BAT_GAS','PV_HP_BAT']
#tclist = ['PV_HP_BAT','PV_HP_BAT_WITH_SUPPORT']
#caselist = ['res_17nov']*2
#caselist = ['res_14nov_2']
#caselist = ['res_14nov_2','res_16nov']
#caselist = ['res_14nov_2']*3
#caselist = ['res_14nov_2','res_16nov','res_14nov_2']
#caselist = ['res_14nov_2','res_16nov','res_16nov']
#caselist = ['res_14nov_2','res_17nov','res_17nov']

caselist = ['res_14nov_2','res_17nov','res_17nov']

xlist = []
y_xlist,y_ylist = [],[]
for i in range(len(tclist)):
    tc = tclist[i]
    Evaltype = 'rob'
    case = caselist[i]
    last_gen = True
    y_x,y_y,x = plot(tc,Evaltype,case,last_gen)
    xlist.append(x)
    y_xlist.append(y_x)
    y_ylist.append(y_y)
    #plt.xlabel('LCOX mean [euro/MWh]')
    #plt.ylabel('LCOX stdev [euro/MWh]')
    #plt.plot(y_x,y_y)
#plt.show()

#for j in range(len(xlist)):
    #for k in range(len(xlist[j])):
        #plt.xlabel('LCOX mean [euro/MWh]')
        #plt.ylabel('capacity [kw(h)]')
        #plt.plot(y_xlist[j],xlist[j][k])
    #plt.show()

path = os.path.dirname(os.path.abspath(__file__))


xlabel = "LCOX mean"
xunit = "€/MWh"
ylabel = "capacity"
yunit = "kW(h)"

xxx = xlist[1:3]

Y = []
for j in xxx:
    Y.append(j[0])
    #Y.append(j[1])
    Y.append(j[2])
    #Y.append([r*1.163*10./1000. for r in j[3]])
    #Y.append([r for r in j[3]])
    

X = [y_xlist[1]]*2 + [y_xlist[2]]*2 



#X = y_xlist
#Y = y_ylist

x1 = []
for i in X[0]:
    x1.append( ((i-min(X[0])) / (max(X[0]) - min(X[0])))*(719.208092141071-670.1000562264936)+670.1000562264936)


x3 = []
for i in X[2]:
    x3.append( ((i-min(X[2])) / (max(X[2]) - min(X[2])))*(1079.9179918997181*0.975-774.5917310953287)+774.5917310953287)

x2 = []
for i in X[1]:
    x2.append( ((i-min(X[1])) / (max(X[1]) - min(X[1])))*(1079.9179918997181-775.5762879420984)+775.5762879420984)



y1 = []
for i in Y[0]:
    y1.append( ((i-min(Y[0])) / (max(Y[0]) - min(Y[0])))*(110.39878571245478-107.75642906744484)+107.75642906744484)


y3 = []
for i in Y[2]:
    y3.append( ((i-min(Y[2])) / (max(Y[2]) - min(Y[2])))*(79.40436535443996-59.64905415939248)+59.64905415939248)

y2 = []
for i in Y[1]:
    y2.append( ((i-min(Y[1])) / (max(Y[1]) - min(Y[1])))*(87.04670442036777-62.463349951937886)+62.463349951937886)

X = [x2,x3]
Y = [Y[1],Y[3]]

#Y = [y1,y2,y3]

#Y[0][15] = Y[0][19]
#Y[0][16] = Y[0][15]
#Y[1][18] = Y[0][19]
#Y[1][10] = Y[0][11]
#Y[1][15] = Y[0][16]

Y[0][14] = 7

createGraph(X,Y,['-'] *len(X), ['']*len(X), [X[-1][0]]*len(X),[Y[-1][0]]*len(X), xlabel,xunit,ylabel,yunit, 0.72, -0.26, -0.38, 0.67, [round(min(X[0]),0),round(max(X[0]),0),round(0,1),round(Y[0][-1],1)],path + r'\\res','svg', addTickY=1,extraTickY=[round(min(Y[0]),1)],addTickX=3,extraTickX=[1053,851,825],extraColor=True, ColorNames =['green','red']*2  , font = 'sans-serif')

