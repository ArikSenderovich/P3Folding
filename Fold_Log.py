from __future__ import division
import csv, pyodbc
from datetime import *
from os.path import exists
import numpy
import datetime
from pytz import *
from pulp import *
from collections import namedtuple
from decimal import Decimal
import numpy
from scipy.stats import norm
import math
from copy import deepcopy
from collections import OrderedDict
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from operator import itemgetter
import xml.etree.ElementTree as ET
import pylab

def HT(n,m,c_a, c_s, rho_, tau):
    EW = numpy.zeros(n)
    for i in range(0,n):
        EW[i] = tau[i]*(c_a[i]+c_s[i])*rho_[i]/(2*m[i]*(1-rho_[i]))

    return EW


def Sak(n,m,c_a, c_s, rho_, tau):
    EW = numpy.zeros(n)
    for i in range(0,n):
        EW[i] = tau[i]*pow(rho_[i],pow(2*(m[i]+1),0.5)-1)*(c_a[i]+c_s[i])/(2*m[i]*(1-rho_[i]))
    return EW


def HalfWhitt(n,m,c_a, c_s, rho_, tau):
    EW = numpy.zeros(n)


    for i in range(0,n):
        beta = pow(m[i],0.5)*(1-rho_[i])
        Pw = 1/(1+pow(2*math.pi,0.5)*beta*norm.cdf(beta)*pow(math.e,pow(beta,2)/2) )
        EW[i] = tau[i]*Pw/(m[i]*(1-rho_[i]))*(c_a[i]+c_s[i])/2
    return EW


def FluidWait(n,m,c_a, c_s, rho_, tau, lam, TimeInterval):
    EW = numpy.zeros(n)
    Horizon = TimeInterval[1] - TimeInterval[0]
    for i in range(0,n):
        if rho_[i]<1:
           EW[i] = tau[i]*pow(rho_[i],pow(2*(m[i]+1),0.5)-1)*(c_a[i]+c_s[i])/(2*m[i]*(1-rho_[i]))
           #EW[i] = 0
        else:
            #Fluid Approx to Q per hour

            Q = (lam[i]-m[i]/tau[i])
            #From little's formula:
            EW[i] = Q/lam[i]





    return EW

def Sojourn(n,lam, ext, EW):

    ET = 0
    EV = numpy.zeros(n)
    for i in range(0,n):
        EV[i] = lam[i]/sum(ext)
        ET+= EV[i] * (tau[i] +EW[i])
    return ET

def QNA_Fluid(n, m, ext, c_ext, tau, c_s, Q_mat, delta_, TimeInterval):


    #Calculate traffic equations (19) - QNA paper
    I = numpy.identity(n)
    Gam = numpy.identity(n)
    for i in range(0,n):
        Gam[i][i] = delta_[i]
    Res = numpy.mat(Gam)*numpy.mat(Q_mat)
    #Res=numpy.mul(Gam,Q_mat)
    #Res = numpy.inner(Gam,Q_mat)
    A = I - Res
    #lam = numpy.linalg.solve(A, ext)
    lam = (ext*(numpy.linalg.inv(A))).tolist()
    lam = lam[0]
    rho_ = numpy.zeros(n)
   # Stability = True
    for i in range(0,n):
                rho_[i] = lam[i] * tau[i] / m[i]
    #Calculating lambda_ij
    #c_sq = c_ext
    c_sq =[]
    for i in range(0,n):
        if lam[i]>0:
            c_sq.append(1)
        else:
            c_sq.append(0)

    #EW = HT(n,m,c_sq,c_s,rho_, tau)

    #EW2 = Sak(n,m,c_sq,c_s,rho_, tau)
    #EW3 = HalfWhitt(n,m,c_sq,c_s,rho_, tau)

    EW = FluidWait(n,m,c_sq, c_s, rho_, tau, lam, TimeInterval)
    soj= Sojourn(n,lam,ext, EW)
    result = []
    result.append(soj)
    result.append(lam)
    result.append(c_sq)

    return result


def QNA(n, m, ext, c_ext, tau, c_s, Q_mat, delta_, TimeInterval):


    #Calculate traffic equations (19) - QNA paper
    I = numpy.identity(n)
    Gam = numpy.identity(n)
    for i in range(0,n):
        Gam[i][i] = delta_[i]
    Res = numpy.mat(Gam)*numpy.mat(Q_mat)
    #Res=numpy.mul(Gam,Q_mat)
    #Res = numpy.inner(Gam,Q_mat)
    A = I - Res
    #lam = numpy.linalg.solve(A, ext)
    lam = (numpy.array(ext)*(numpy.linalg.inv(A))).tolist()
    lam = lam[0]


    Stability = True

    #lam = numpy.linalg.solve(A, ext)
    #lam2 = numpy.inner(ext,numpy.linalg.inv(Res))
    #print(numpy.allclose(numpy.dot(Gam, lam), Q_mat))

    #Calculating the utilization per node
    rho_ = numpy.zeros(n)
    for i in range(0,n):
            if lam[i]*tau[i]>=m[i]:
                #Infinite Server approximation
                rho_[i] = 0
                Stability = False

            else:
                rho_[i] = lam[i] * tau[i] / m[i]

    #Calculating lambda_ij

    lambda_ij = numpy.identity(n)
    p=numpy.identity(n)
    for i in range(0,n):
        for j in range(0,n):
            lambda_ij[i][j] = lam[i]*delta_[i]*Q_mat[i][j]
            p[i][j] = lambda_ij[i][j]/lam[j]
    p_ext = numpy.zeros(n)
    for j in range(0,n):
        p_ext[j]= ext[j]/lam[j]


    #calculating x, v, w
    x = numpy.zeros(n)
    for i in range(0,n):
        x[i]= 1+pow(m[i],0.5)*(max(c_s[i],0.2) -1)
    v = numpy.identity(n)
    temp_sum = 0
    for j in range(0,n):
        for i in range(0,n):
            temp_sum += pow(p[i][j],2)
        temp_sum+= pow(p_ext[j],2)
        v[j][j]=1/temp_sum
        temp_sum =0

    w = numpy.zeros(n)
    for i in range(0,n):
        w[i]= pow((1+4*pow(1-rho_[i],2)*(v[i][i]-1)),-1)

    #a, b:
    a =numpy.zeros(n)
    temp_sum = 0
    for j in range(0,n):
        for i in range(0,n):
            temp_sum+= p[i][j] *(1-Q_mat[i][j]) + (1-v[i][j])*delta_[i]*Q_mat[i][j] * pow(rho_[i],2) * x[i]
        a[j] = 1+ w[j] *((p_ext[j] * c_ext[j] -1) +temp_sum )



    b =numpy.identity(n)
    for i in range(0,n):
        for j in range(0,n):
            b[i][j] = w[j]* p[i][j]* Q_mat[i][j]* delta_[i]*(v[i][j] + (1-v[i][j])*(1-pow(rho_[i],2)))
    Res = I * numpy.mat(b)
    A = I - Res

    #Decomposition to stations (c_aj)
    c_sq = (a*numpy.linalg.inv(A)).tolist()
    c_sq = c_sq[0]


    #Now, calculate average waiting per node, then combine into network approx.
    EW = FluidWait(n,m,c_sq, c_s, rho_, tau, lam, TimeInterval)
    #EW1 = HT(n,m,c_sq,c_s,rho_, tau)
    EW2 = Sak(n,m,c_sq,c_s,rho_, tau)
    EW3 = HalfWhitt(n,m,c_sq,c_s,rho_, tau)
    soj= Sojourn(n,lam,ext, EW)
    result = []
    result.append(soj)
    result.append(lam)
    result.append(c_sq)
    result.append(Stability)
    if Stability ==False:
        print('Bullshit!')
    return result

def QNA_Arik(n, m, ext, c_ext, tau, c_s, Q_mat, delta_):


    #Calculate traffic equations (19) - QNA paper
    I = numpy.identity(n)
    Gam = numpy.identity(n)
    for i in range(0,n):
        Gam[i][i] = delta_[i]
    Res = numpy.mat(Gam)*numpy.mat(Q_mat)
    #Res=numpy.mul(Gam,Q_mat)
    #Res = numpy.inner(Gam,Q_mat)
    A = I - Res
    #lam = numpy.linalg.solve(A, ext)
    lam = (ext*(numpy.linalg.inv(A))).tolist()
    lam = lam[0]
    rho_ = numpy.zeros(n)
    Stability = True
    for i in range(0,n):

            if lam[i]*tau[i]>=m[i]:
                #Infinite Server approximations
                rho_[i] = 0
                Stability = False
            else:
                rho_[i] = lam[i] * tau[i] / m[i]


    #Calculating lambda_ij
    #c_sq = c_ext
    c_sq = [1 for i in range(0,n)]

    EW1 = HT(n,m,c_sq,c_s,rho_, tau)
    EW2 = Sak(n,m,c_sq,c_s,rho_, tau)
    EW3 = HalfWhitt(n,m,c_sq,c_s,rho_, tau)


    soj= Sojourn(n,lam,ext, EW1)
    result = []
    result.append(soj)
    result.append(lam)
    result.append(c_sq)
    result.append(Stability)
    if Stability ==False:
        print('Bullshit!')
    return result


def ReadTreeTuple(path):
    tree = ET.parse(path)
    root = tree.getroot()

   # for neighbor in root.iter('neighbor'):

    nodes_ = []
    node_ =[]

    for child in root:
        for k in child:
           if k.tag == "manualTask":
            node_.append("Transition")
            node_.append(k.attrib.get('id'))
            node_.append(k.attrib.get('name'))
            nodes_.append(node_)
            node_=[]

           elif k.tag!="automaticTask" and k.tag!="parentsNode":
            node_.append(k.tag)
            node_.append(k.attrib.get('id'))
            nodes_.append(node_)
            node_=[]

    FoldTree= []
    #TreeNode = []

    TreeNode = namedtuple('TreeNode', 'id, old_id, type, name, parents, children, weights, service,m,visit_xor, folded, fold_id, fold_prev, arrival_rate,c_sq')
    pairs={}
    node_id = 0

    for n in nodes_:
        #New id:

        TreeNode.id = node_id
        #Old id:
        TreeNode.old_id = n[1]

        pairs[n[1]] = node_id
        #Type:
        TreeNode.type = n[0]
        #Name for transitions
        if n[0] == "Transition":
            TreeNode.type = "Activity"
            TreeNode.name = n[2]
            TreeNode.folded =1
            TreeNode.fold_id =0
            TreeNode.fold_prev  =[]
        else:
            TreeNode.type ="Operator"
            TreeNode.name  = n[0]
            TreeNode.folded = 0
            TreeNode.fold_id =0
            TreeNode.fold_prev  =[]



        FoldTree.append(TreeNode)
        #For xorLoop, weights consist of XOR weights and last value is probability to repeat the structure
        TreeNode = namedtuple('TreeNode', 'id, old_id, type, name, parents, children, weights, service, m, visit_xor, folded, fold_id fold_prev,arrival_rate,c_sq')
        node_id+=1

    links = []
    link = []

    for child in root:
        for k in child:
            if k.tag =="parentsNode":
               source =k.attrib.get('sourceId')
               dest = k.attrib.get('targetId')
               try:
                   print(pairs[source])
                   print(pairs[dest])
                   link.append(source)
                   link.append(dest)
                   links.append(link)
                   link=[]
               except KeyError:
                   print('skipping' + source + " to "+ dest)

    children=[]
    parents = []
    for n in FoldTree:
       for l in links:
                #gathering all children
                if n.old_id == l[0]:
                    children.append(pairs[l[1]])
                #gathering all parents
                if n.old_id == l[1]:
                    parents.append(pairs[l[0]])
       n.parents = parents
       n.children = children

       children =[]
       parents = []

    return FoldTree

#ExtractLogDFCI()
def getLog(path):

    Log = []
    trace = []
    with open(path, 'rb') as csvfile:
                reader_ = csv.reader(csvfile, delimiter=',')
                next(reader_, None)
                for r in reader_:
                    #id
                    trace.append(r[0])
                    #resource
                    trace.append(r[1])
                    #start time
                    trace.append(r[2])
                    #end time
                    trace.append(r[3])
                    Log.append(trace)
                    trace=[]
    return Log
def tracifyLog(Log):
    cur_id = Log[0][0]
    trace = []
    trace_log = []

    for t in Log:
        if t[0]!=cur_id:
            cur_id = t[0]
            trace_log.append(trace)
            trace=[]
            trace.append(t[1])
        else:
            trace.append(t[1])
    return trace_log

def RemoveOutliers(sample,sigmas):
    if len(sample) ==0:
        return -1
    else:
            mean_ = sum(sample)/(len(sample))

            sum_ = 0
            for i in sample:
                    sum_ += pow((i - mean_),2)
            std= round(float(pow(sum_/(len(sample)-1),0.5)),2)
            final_sample = []
            for i in sample:
                if i < mean_+sigmas*std or i > mean_-sigmas*std:
                    final_sample.append(i)


    return final_sample

def NumServers(Name, Log, TimeInterval):
    Log = sorted(Log, key = itemgetter(2,3))


    tmp = datetime.datetime.strptime(Log[0][2],'%Y/%m/%d %H:%M:%S.%f')
    end_int = datetime.datetime(tmp.year, tmp.month,tmp.day,hour = TimeInterval[1],minute=0,second=0)
    cur_day = end_int.day
    events =[]
    event = []
    max_s = []
    for l in Log:
        if l[1] == Name:
            start = datetime.datetime.strptime(l[2],'%Y/%m/%d %H:%M:%S.%f')
            end = datetime.datetime.strptime(l[3],'%Y/%m/%d %H:%M:%S.%f')
            if start<end_int:
                event.append(start)
                event.append(1)
                events.append(event)
                event = []
            if end<end_int:
                event.append(end)
                event.append(-1)
                events.append(event)
                event = []

            if cur_day != start.day:
                events = sorted(events,key=itemgetter(0))
                max_ = 0
                count_ = 0
                for e in events:
                        if e[1]==1:
                            count_+=1
                            if count_>max_:
                                max_=count_
                        else:
                            count_-=1

                if max_>0:
                    max_s.append(max_)
                cur_day = start.day
                events = []
                end_int = datetime.datetime(end.year, end.month,end.day,hour = TimeInterval[1],minute=0,second=0)
    #print(max_s)




    return round(float(sum(max_s))/float(len(max_s)),2)
#Expects to have Arrivals activity in the log
def Lambda(Log, TimeInterval):
    count = 0
    cur_day = datetime.datetime.strptime(Log[0][3],'%Y/%m/%d %H:%M:%S.%f').day
    days = 1
    for l in Log:
        if l[1] == "Arrival":
            end = datetime.datetime.strptime(l[3],'%Y/%m/%d %H:%M:%S.%f')
            start = datetime.datetime.strptime(l[2],'%Y/%m/%d %H:%M:%S.%f')
            if start.day !=cur_day:
                days+=1
                cur_day = start.day
            hr_end = end.hour
            hr_start = start.hour


            if (hr_start >= TimeInterval[0] and hr_end<TimeInterval[1]):
                count+=1

    return (count)/((TimeInterval[1]-TimeInterval[0])*days)
def ExpectedService(Name,Log):

    sample =[]
    for l in Log:
        if l[1] ==Name:
            end = datetime.datetime.strptime(l[3],'%Y/%m/%d %H:%M:%S.%f')
            start = datetime.datetime.strptime(l[2],'%Y/%m/%d %H:%M:%S.%f')
            elapsedTime = end-start
            sample.append(divmod(elapsedTime.total_seconds(), 60)[0]*60+divmod(elapsedTime.total_seconds(), 60)[1])

    sample = RemoveOutliers(sample,2)

    return sum(sample)/len(sample)

def ReturnSuccessors(id, FoldTree,succ):
    #succ= []
    print (str(FoldTree[id].name))
    for c in FoldTree[id].children:
        print (str(FoldTree[c].name))
    if len(FoldTree[id].children)==0:
        return FoldTree[id].name
    else:
        #succ =[]
        for c in FoldTree[id].children:
            succ.append(ReturnSuccessors(c,FoldTree,succ))

    return succ

def ReturnSuccessorsNew(id, FoldTree,succ):

    if len(FoldTree[id].children)==0:
        succ.append(FoldTree[id].name)
        return
    else:
        #succ =[]
        for c in FoldTree[id].children:
            ReturnSuccessorsNew(c,FoldTree,succ)



    return succ


def ProportionsXor(p, TraceLog):
    prop = 0
    count_traces = 0
    for t in TraceLog:
        if len(set(t).intersection(p))>0:
            count_traces+=1


    return count_traces/len(TraceLog)

def RepeatProb(name, trace_log):

    prob_ = []
    appear_ = []
    for t in trace_log:
        count_rep =0
        for k in t:
            if name==k:
                count_rep+=1
        if count_rep>0:
            prob_.append((count_rep-1)/(count_rep))


    return sum(prob_)/len(prob_)


def EnrichTree(FoldTree,Log,TraceLog,TimeInterval):

    for n in FoldTree:
        if n.type == "Activity":
            if n.name != "Arrival":
                serv = ExpectedService(n.name,Log)
                m_  = NumServers(n.name, Log, TimeInterval)
                n.service = serv
                n.m = m_
            else:
                lambda_ = Lambda(Log, TimeInterval)
                #This is not service time, but the arrival rate
                n.arrival_rate = lambda_
                n.service = 0
                n.m = 100000
                n.c_sq = 1
        else:
            if n.name == 'xorLoop':
                n.weights = round(RepeatProb(FoldTree[n.children[0]].name,TraceLog),2)

    return FoldTree


def ReturnIntersect(p, log):
    new_log = []
    for t in log:
        if len(set(t).intersection(p))>0:
            new_log.append(t)


    return new_log

def NextXor(id, FoldTree, TraceLog):
     if FoldTree[id].name=='xor' and FoldTree[id].visit==0:
        children = FoldTree[id].children
        weights = []
        num_children = len(children)+1
        for i in range(0,num_children):
            weights.append(1/num_children)

        partition =[]
        for child in children:
            succ = []

            succ = ReturnSuccessorsNew(child,FoldTree,succ)
            if succ == None:
                succ= []

            if FoldTree[child].type == "Activity":
                succ.append(FoldTree[child].name)
            partition.append(succ)
        sum_=0


        for i,p in enumerate(partition):
            weights[i] = round(float(ProportionsXor(p,TraceLog)),2)
            sum_+=weights[i]

        #todo: bug! weights for the xor sum to more than 1 if skip is allowed benieth - taus should be implemented as well

        weights[len(weights)-1] =1-sum_
        FoldTree[id].weights = weights
        FoldTree[id].visit = 1
        for p in partition:
            new_log = ReturnIntersect(p,TraceLog)
            NextXor(id,FoldTree,new_log)
        return

     elif len(FoldTree[id].children)==0:
         return

     else:
        #succ =[]
        for c in FoldTree[id].children:
            NextXor(c,FoldTree,TraceLog)



     return FoldTree

def ConnectTreeTraceLog(FoldTree, TraceLog):
    root_id = 0
    for n in FoldTree:
        if len(n.parents)==0:
            root_id = n.id
        n.visit = 0


    ConnectedTree = NextXor(root_id,FoldTree,TraceLog)


    return ConnectedTree

def LocateNodeInG(node_id,G):
    for n in G:
        if node_id == n.tree_id:
           return n
    return -1

def toGraph(id,FoldTree,G):
    if FoldTree[id].type =="Activity":
        return
    else:
        #Unfolding sequence
        if FoldTree[id].name == "sequence":
            #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
            #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound


            Tree_Children = FoldTree[id].children
            Left = Tree_Children[0]
            Right = Tree_Children[len(Tree_Children)-1]

            #Leftmost node of a sequence
            Node_Left = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_Left.repeat =0
            Node_Left.tree_id = FoldTree[Left].id
            Node_Left.name = FoldTree[Left].name
            Node_Left.type = FoldTree[Left].type
            Node_Left.inbound = in_
            Node_Left.outbound = [Tree_Children[1]]
            if Node_Left.name == 'xor':
                Node_Left.out_weights = FoldTree[Left].weights
            elif Node_Left.name == 'xorLoop':
                Node_Left.repeat = FoldTree[Left].weights



            #Rightmost node of a sequence
            Node_Right = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_Right.repeat =0
            Node_Right.tree_id = FoldTree[Right].id
            Node_Right.name = FoldTree[Right].name
            Node_Right.type = FoldTree[Right].type
            Node_Right.inbound = [Tree_Children[len(Tree_Children)-2]]
            Node_Right.outbound = out
            if Node_Right.name == 'xor':
                Node_Right.out_weights = FoldTree[Right].weights
            elif Node_Right.name == 'xorLoop':
                Node_Right.repeat = FoldTree[Right].weights


            if len(in_)>0:
                for i in in_:
                    pred = LocateNodeInG(i,G)
                    pred.outbound.append(FoldTree[Left].id)
                    pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                for o in out:
                    succ = LocateNodeInG(o,G)
                    succ.inbound.append(FoldTree[Right].id)
                    succ.inbound.remove(FoldTree[id].id)
            G.append(Node_Left)
            G.append(Node_Right)

            Temp = []
            for t in Tree_Children:
                Temp.append(t)
            Tree_Children.remove(Left)
            Tree_Children.remove(Right)
            if len(Tree_Children)>0:
                for c in Tree_Children:
                    Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                    Node_New.repeat = 0
                    Node_New.tree_id = FoldTree[c].id
                    Node_New.name = FoldTree[c].name
                    Node_New.type = FoldTree[c].type
                    prev_ind = Temp.index(c)-1
                    next_ind = Temp.index(c)+1
                    Node_New.inbound = [Temp[prev_ind]]
                    Node_New.outbound = [Temp[next_ind]]
                    if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                    elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights
                    G.append(Node_New)
            G.remove(v_source)
            FoldTree[id].children = []

            for c in Temp:
                FoldTree[id].children.append(c)
                toGraph(c,FoldTree,G)
            #Unfolding xorLoop
            #Currently, xorLoop is supported only for direct activities being successors (at most one?)
        elif FoldTree[id].name == "xorLoop":
            #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
            #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound
            repeat_ = v_source.repeat
            c = FoldTree[id].children[0]

            Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_New.repeat = 0
            Node_New.tree_id = FoldTree[c].id
            Node_New.name = FoldTree[c].name
            Node_New.type = FoldTree[c].type
            Node_New.inbound = in_
            Node_New.outbound = out
            Node_New.repeat = repeat_
            if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(Node_New.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(Node_New.tree_id)
                        succ.inbound.remove(FoldTree[id].id)
            G.append(Node_New)
            G.remove(v_source)

        elif FoldTree[id].name == "xor":
            #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
            #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound
            weights = v_source.out_weights
            Tree_Children = FoldTree[id].children
            before = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            before.repeat = 0
            before.tree_id = FoldTree[id].id
            before.name = FoldTree[id].name+'_Before'
            before.type = FoldTree[id].type
            before.inbound = in_
            before.weights = weights
            before.outbound = []

            after = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            after.repeat = 0
            after.tree_id = FoldTree[id].id+1000
            after.name = FoldTree[id].name+'_After'
            after.type = FoldTree[id].type
            after.outbound = out
            after.inbound =[]

            G.append(before)
            G.append(after)

            if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(before.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(after.tree_id)
                        succ.inbound.remove(FoldTree[id].id)
            G.remove(v_source)


            for c in Tree_Children:
                Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                Node_New.repeat = 0
                Node_New.tree_id = FoldTree[c].id
                Node_New.name = FoldTree[c].name
                Node_New.type = FoldTree[c].type
                Node_New.inbound = [before.tree_id]
                Node_New.outbound = [after.tree_id]
                if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights

                after.inbound.append(FoldTree[c].id)
                before.outbound.append(FoldTree[c].id)


                G.append(Node_New)

                toGraph(c,FoldTree,G)

            before.outbound.append(after.tree_id)






        elif FoldTree[id].name == "and":
                #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
                #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
                v_source = LocateNodeInG(id,G)
                out = v_source.outbound
                in_ = v_source.inbound
                Tree_Children = FoldTree[id].children
                before = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                before.repeat = 0
                before.tree_id = FoldTree[id].id
                before.name = FoldTree[id].name+'_Before'
                before.type = FoldTree[id].type
                before.inbound = in_
                before.outbound = []

                after = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                after.repeat = 0
                after.tree_id = FoldTree[id].id+1000
                after.name = FoldTree[id].name+'_After'
                after.type = FoldTree[id].type
                after.outbound = out
                after.inbound =[]

                if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(before.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

                if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(after.tree_id)
                        succ.inbound.remove(FoldTree[id].id)

                G.append(before)
                G.append(after)
                G.remove(v_source)
                for c in Tree_Children:
                    Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                    Node_New.repeat = 0
                    Node_New.tree_id = FoldTree[c].id
                    Node_New.name = FoldTree[c].name
                    Node_New.type = FoldTree[c].type
                    Node_New.inbound = [before.tree_id]
                    Node_New.outbound = [after.tree_id]
                    if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                    elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights
                    after.inbound.append(FoldTree[c].id)
                    before.outbound.append(FoldTree[c].id)
                    G.append(Node_New)
                    toGraph(c,FoldTree,G)



    return
def toGraphFold(id,FoldTree,G, Foldings):
    if FoldTree[id].type =="Activity" or FoldTree[id].fold_id in Foldings:
        return
    else:
        #Unfolding sequence
        if FoldTree[id].name == "sequence":

            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound


            Tree_Children = FoldTree[id].children
            Left = Tree_Children[0]
            Right = Tree_Children[len(Tree_Children)-1]

            #Leftmost node of a sequence
            Node_Left = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_Left.repeat =0
            Node_Left.tree_id = FoldTree[Left].id
            Node_Left.name = FoldTree[Left].name
            if FoldTree[Left].fold_id in Foldings:
                Node_Left.type = "Activity"
            else:
                Node_Left.type = FoldTree[Left].type
            Node_Left.inbound = in_
            Node_Left.outbound = [Tree_Children[1]]
            if Node_Left.name == 'xor':
                Node_Left.out_weights = FoldTree[Left].weights
            elif Node_Left.name == 'xorLoop':
                Node_Left.repeat = FoldTree[Left].weights



            #Rightmost node of a sequence
            Node_Right = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_Right.repeat =0
            Node_Right.tree_id = FoldTree[Right].id
            Node_Right.name = FoldTree[Right].name
            if FoldTree[Right].fold_id in Foldings:
                Node_Right.type = "Activity"
            else:
                Node_Right.type = FoldTree[Right].type
            Node_Right.inbound = [Tree_Children[len(Tree_Children)-2]]
            Node_Right.outbound = out
            if Node_Right.name == 'xor':
                Node_Right.out_weights = FoldTree[Right].weights
            elif Node_Right.name == 'xorLoop':
                Node_Right.repeat = FoldTree[Right].weights


            if len(in_)>0:
                for i in in_:
                    pred = LocateNodeInG(i,G)
                    pred.outbound.append(FoldTree[Left].id)
                    pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                for o in out:
                    succ = LocateNodeInG(o,G)
                    succ.inbound.append(FoldTree[Right].id)
                    succ.inbound.remove(FoldTree[id].id)
            G.append(Node_Left)
            G.append(Node_Right)

            Temp = []
            for t in Tree_Children:
                Temp.append(t)
            Tree_Children.remove(Left)
            Tree_Children.remove(Right)
            if len(Tree_Children)>0:
                for c in Tree_Children:
                    Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                    Node_New.repeat = 0
                    Node_New.tree_id = FoldTree[c].id
                    Node_New.name = FoldTree[c].name
                    if FoldTree[c].fold_id in Foldings:
                        Node_New.type = "Activity"
                    else:
                        Node_New.type = FoldTree[c].type
                    prev_ind = Temp.index(c)-1
                    next_ind = Temp.index(c)+1
                    Node_New.inbound = [Temp[prev_ind]]
                    Node_New.outbound = [Temp[next_ind]]
                    if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                    elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights
                    G.append(Node_New)
            G.remove(v_source)
            FoldTree[id].children = []

            for c in Temp:
                FoldTree[id].children.append(c)
                toGraphFold(c,FoldTree,G,Foldings)
            #Unfolding xorLoop
            #Currently, xorLoop is supported only for direct activities being successors (at most one?)
        elif FoldTree[id].name == "xorLoop":
            #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
            #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound
            repeat_ = v_source.repeat
            c = FoldTree[id].children[0]

            Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            Node_New.repeat = 0
            Node_New.tree_id = FoldTree[c].id
            Node_New.name = FoldTree[c].name
            Node_New.type = FoldTree[c].type
            Node_New.inbound = in_
            Node_New.outbound = out
            Node_New.repeat = repeat_
            if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(Node_New.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(Node_New.tree_id)
                        succ.inbound.remove(FoldTree[id].id)
            G.append(Node_New)
            G.remove(v_source)

        elif FoldTree[id].name == "xor":
            #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
            #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
            v_source = LocateNodeInG(id,G)
            out = v_source.outbound
            in_ = v_source.inbound
            weights = v_source.out_weights
            Tree_Children = FoldTree[id].children
            before = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            before.repeat = 0
            before.tree_id = FoldTree[id].id
            before.name = FoldTree[id].name+'_Before'
            before.type = FoldTree[id].type
            before.inbound = in_
            before.weights = weights
            before.outbound = []

            after = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
            after.repeat = 0
            after.tree_id = FoldTree[id].id+1000
            after.name = FoldTree[id].name+'_After'
            after.type = FoldTree[id].type
            after.outbound = out
            after.inbound =[]

            G.append(before)
            G.append(after)

            if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(before.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

            if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(after.tree_id)
                        succ.inbound.remove(FoldTree[id].id)
            G.remove(v_source)


            for c in Tree_Children:
                Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                Node_New.repeat = 0
                Node_New.tree_id = FoldTree[c].id
                Node_New.name = FoldTree[c].name
                Node_New.type = FoldTree[c].type
                Node_New.inbound = [before.tree_id]
                Node_New.outbound = [after.tree_id]
                if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights

                after.inbound.append(FoldTree[c].id)
                before.outbound.append(FoldTree[c].id)


                G.append(Node_New)

                toGraphFold(c,FoldTree,G,Foldings)

            before.outbound.append(after.tree_id)






        elif FoldTree[id].name == "and":
                #Node = namedtuple('GraphNode', 'tree_id, name , type, final')
                #Edge = namedtuple('Edge', 'source_id','dest_id', 'weight')
                v_source = LocateNodeInG(id,G)
                out = v_source.outbound
                in_ = v_source.inbound
                Tree_Children = FoldTree[id].children
                before = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                before.repeat = 0
                before.tree_id = FoldTree[id].id
                before.name = FoldTree[id].name+'_Before'
                before.type = FoldTree[id].type
                before.inbound = in_
                before.outbound = []

                after = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                after.repeat = 0
                after.tree_id = FoldTree[id].id+1000
                after.name = FoldTree[id].name+'_After'
                after.type = FoldTree[id].type
                after.outbound = out
                after.inbound =[]

                if len(in_)>0:
                    for i in in_:
                        pred = LocateNodeInG(i,G)
                        pred.outbound.append(before.tree_id)
                        pred.outbound.remove(FoldTree[id].id)

                if len(out)>0:
                    for o in out:
                        succ = LocateNodeInG(o,G)
                        succ.inbound.append(after.tree_id)
                        succ.inbound.remove(FoldTree[id].id)

                G.append(before)
                G.append(after)
                G.remove(v_source)
                for c in Tree_Children:
                    Node_New = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
                    Node_New.repeat = 0
                    Node_New.tree_id = FoldTree[c].id
                    Node_New.name = FoldTree[c].name
                    if FoldTree[c].fold_id in Foldings:
                        Node_New.type = "Activity"
                    else:
                        Node_New.type = FoldTree[c].type
                    Node_New.inbound = [before.tree_id]
                    Node_New.outbound = [after.tree_id]
                    if Node_New.name == 'xor':
                        Node_New.out_weights = FoldTree[c].weights
                    elif Node_New.name == 'xorLoop':
                        Node_New.repeat = FoldTree[c].weights
                    after.inbound.append(FoldTree[c].id)
                    before.outbound.append(FoldTree[c].id)
                    G.append(Node_New)
                    toGraphFold(c,FoldTree,G,Foldings)



    return

def UnFoldQNANew(FoldTree, Foldings):

    G = []
    #E = []
    #Type: AND, XOR, Activity, RepeatActivity; Final=0 (not final), 1 (final)
    Node = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
    #Edge = namedtuple('Edge', 'source_id,dest_id ,weight')
    root_id = 0

    for n in FoldTree:
        if len(n.parents)==0:
            root_id = n.id
        n.visit = 0
    Node.tree_id = root_id
    Node.name = FoldTree[root_id].name
    Node.type = FoldTree[root_id].type
    Node.inbound = []
    Node.outbound = []
    Node.out_weights = []
    Node.repeat =[]
    G.append(Node)
    if FoldTree[root_id].type == "Activity":
        return -1


    toGraphFold(root_id,FoldTree,G, Foldings)




    tau = []
    c_s = []
    m = []
    delta_ = []

    states =[]
    state_names =[]
    arr_ind =0
    lambda_=0
    arriving_id = 0
    for i,g in enumerate(G):
        #states.append(g.tree_id)
        #state_names.append(g.name)
        if g.name=="Arrival":
            #arr_ind = i
            #c_s.append(1)
            #lambda_ = ext.append(FoldTree[g.tree_id].service)
            lambda_ = FoldTree[g.tree_id].arrival_rate
            arriving_id = g.tree_id
            #c_ext.append(1)
            #delta_.append(1)
            #tau.append(0)
            #m.append(100000)
        else:
            states.append(g.tree_id)
            state_names.append(g.name)
            if g.type == "Activity":
                if g.repeat>0:
                    tau.append(float(FoldTree[g.tree_id].service/3600)/(1-g.repeat))
                else:
                    tau.append(float(FoldTree[g.tree_id].service/3600))
                c_s.append(1)
                delta_.append(1)
                m.append(FoldTree[g.tree_id].m)
            else:
                if g.name == "xorLoop":
                    tau.append(float(FoldTree[g.tree_id].service/3600))
                    c_s.append(1)
                    delta_.append(1)
                    m.append(FoldTree[g.tree_id].m)
                else:

                    tau.append(0)
                    c_s.append(1)
                    if g.name=="and_After":
                        delta_.append(1/len(g.inbound))
                    elif g.name =="and_Before":
                        delta_.append(len(g.outbound))
                    else:
                        delta_.append(1)

                    m.append(100000)


    Q_mat = [[0 for i in range(0,len(states))] for j in range(0,len(states))]
    ext = [0 for i in range(0,len(states))]
    c_ext = [0 for i in range(0,len(states))]
    for g in G:
        if arriving_id in g.inbound:
            ind = states.index(g.tree_id)
            ext[ind] = lambda_
            c_ext[ind] =1

    for i,s in enumerate(states):
        v_source = LocateNodeInG(states[i],G)
        if state_names[i] == "xor_Before":
            v_destinations = v_source.outbound
            v_weights = v_source.weights
            for j,v in enumerate(v_destinations):
                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = v_weights[j]
                except ValueError:
                    print ('yuppeee')
        elif state_names[i] =="and_Before":
            v_destinations = v_source.outbound
            for j,v in enumerate(v_destinations):

                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = 1/len(v_destinations)
                except ValueError:
                    print ('yuppeee')


        else:
            v_destinations = v_source.outbound
            for j,v in enumerate(v_destinations):

                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = 1
                except ValueError:
                    print ('yuppeee')


    QNA_par = [len(states),m,ext,c_ext,tau,c_s,Q_mat,delta_,state_names]
    return QNA_par

def UnFoldQNA(FoldTree):

    G = []
    #E = []
    #Type: AND, XOR, Activity, RepeatActivity; Final=0 (not final), 1 (final)
    Node = namedtuple('GraphNode', 'tree_id, name, type, inbound, outbound, out_weights, repeat')
    #Edge = namedtuple('Edge', 'source_id,dest_id ,weight')
    root_id = 0

    for n in FoldTree:
        if len(n.parents)==0:
            root_id = n.id
        n.visit = 0
    Node.tree_id = root_id
    Node.name = FoldTree[root_id].name
    Node.type = FoldTree[root_id].type
    Node.inbound = []
    Node.outbound = []
    Node.out_weights = []
    Node.repeat =[]
    G.append(Node)
    if FoldTree[root_id].type == "Activity":
        return -1

        
    GVE = toGraph(root_id,FoldTree,G)

    #for g in G:
     #   print('Node:')
      #  print(g.tree_id)
      #  print(g.name)
      #  print(g.inbound)
      #  print(g.outbound)

    ext = []
    c_ext = []
    tau = []
    c_s = []
    m = []
    delta_ = []

    states =[]
    state_names =[]
    arr_ind =0
    lambda_=0
    arriving_id = 0
    for i,g in enumerate(G):
        #states.append(g.tree_id)
        #state_names.append(g.name)
        if g.name=="Arrival":
            #arr_ind = i
            #c_s.append(1)
            #lambda_ = ext.append(FoldTree[g.tree_id].service)
            lambda_ = FoldTree[g.tree_id].arrival_rate
            arriving_id = g.tree_id
            #c_ext.append(1)
            #delta_.append(1)
            #tau.append(0)
            #m.append(100000)
        else:
            states.append(g.tree_id)
            state_names.append(g.name)
            if g.type == "Activity":
                if g.repeat>0:
                    tau.append(float(FoldTree[g.tree_id].service/3600)/(1-g.repeat))
                else:
                    tau.append(float(FoldTree[g.tree_id].service/3600))
                c_s.append(1)
                delta_.append(1)
                m.append(FoldTree[g.tree_id].m)
            else:
                tau.append(0)
                c_s.append(1)
                if g.name=="and_After":
                    delta_.append(1/len(g.inbound))
                elif g.name =="and_Before":
                    delta_.append(len(g.outbound))
                else:
                    delta_.append(1)

                m.append(100000)


    Q_mat = [[0 for i in range(0,len(states))] for j in range(0,len(states))]
    ext = [0 for i in range(0,len(states))]
    c_ext = [0 for i in range(0,len(states))]
    for g in G:
        if arriving_id in g.inbound:
            ind = states.index(g.tree_id)
            ext[ind] = lambda_
            c_ext[ind] =1

    for i,s in enumerate(states):
        v_source = LocateNodeInG(states[i],G)
        if state_names[i] == "xor_Before":
            v_destinations = v_source.outbound
            v_weights = v_source.weights
            for j,v in enumerate(v_destinations):
                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = v_weights[j]
                except ValueError:
                    print ('yuppeee')
        elif state_names[i] =="and_Before":
            v_destinations = v_source.outbound
            for j,v in enumerate(v_destinations):

                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = 1/len(v_destinations)
                except ValueError:
                    print ('yuppeee')


        # elif state_names[i] =="and_After":
        #     v_sources = v_source.inbound
        #     for j,v in enumerate(v_sources):
        #         v_ = states.index(v)
        #         Q_mat[i][v_] = 1/len(v_sources)
        else:
            v_destinations = v_source.outbound
            for j,v in enumerate(v_destinations):

                try:
                    v_dest = states.index(v)
                    Q_mat[i][v_dest] = 1
                except ValueError:
                    print ('yuppeee')

        # for i,s in enumerate(states):
        #     if state_names[i] == "and_Before":
        #         j_ind =[]
        #         for j in range(0,len(states)):
        #
        #             if Q_mat[i][j]>0:
        #                 j_ind.append(j)
        #         for j in j_ind:
        #             delta_[j] = len(j_ind)
        #

    #for i in range(0,len(states)):
      #  print(sum(Q_mat[i]))

    QNA_par = [len(states),m,ext,c_ext,tau,c_s,Q_mat,delta_,state_names]
    return QNA_par


def encode_ILP(c, b, B, S):
    leaves=len(c)
    leaf_range = range(0,leaves)
    #initiating variables for the ILP
    vars = LpVariable.dicts("x",[i for i in leaf_range],0,1,cat=LpBinary)
    #initiating problem - score function is maximal utility (or value)
    prob = LpProblem("Folding",LpMaximize)
    prob += lpSum([b[i]*vars[i] for i in leaf_range])
    #cost budget <= B
    prob += sum([c[i]*vars[i] for i in leaf_range]) <= B


    #Precedence order:
    for i in leaf_range:
        if len(S[i])>0:
            for j in S[i]:
                c = LpAffineExpression([ (vars[j],1), (vars[i],-1)])
                prob += LpConstraint(c, sense=1, rhs=0)


    prob.writeLP("Folding.lp")
    prob.solve(GUROBI_CMD(msg=0))
    sol = []
    #print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        if v.varValue>0:
            #print(v.name, "=", v.varValue)
            sol.append(int(str(v).split("_")[1]))
    #print("Total Value: ", value(prob.objective))

    return(sol)

def CalcXorCost(weights, tau, m, lam,TimeInterval):
    res = 0
    n = len(weights)
    q_prob = []
    delta_=[]
    m.append(100000)
    lam.append(0)
    tau.append(0)
    for i in range(0,n):
        if i<n-1:
            q_prob.append(weights[i])
        else:
            q_prob.append(0)
        delta_.append(1)
    c_a = [1 for i in range(0,n-1)]
    c_s = [1 for i in range(0,n)]
    c_a.append(0)


    Q_mat = [[0 for i in range(0,n)] for i in range(0,n)]
    for i in range(0,n-1):
        Q_mat[i][n-1] = 1


    before = QNA_Fluid(n,m,lam,c_a,tau,c_s,Q_mat,delta_, DayTimeInterval)

    n = 2

    m = [sum(m[0:(len(m)-1)]),100000]
    #m=[1,10000]
    #m =[m_,100000]
    lam = [sum(lam),0]
    delta_ = [1,1]
    t_ = 0
    for i,t in enumerate(tau):
        t_ += t*weights[i]
    #m =[1,100000]
    tau = [t_,0]
    c_a = [1,0]
    c_s = [1,1]
    Q_mat = [[0,1],[0,0]]
    after = QNA_Fluid(n,m,lam,c_a,tau,c_s,Q_mat,delta_, TimeInterval)
    res =(after[0]-before[0])*60
    return res
def CalcAndCost(tau, m, lam,TimeInterval):
    res = 0
    n = len(tau)
    q_prob = []
    delta_=[]

    m.insert(0, 100000)
    ext = [0 for i in range(0,n)]
    ext.insert(0,lam)
    tau.insert(0,0)
    #lam.append(0)
    #tau.append(0)
    delta_.append(2)
    for i in range(0,n):
        delta_.append(1)

    c_a = [0 for i in range(0,n)]
    c_a.insert(0,1)
    c_s = [1 for i in range(0,n+1)]
    #c_a.append(0)


    Q_mat = [[0 for i in range(0,n+1)] for i in range(0,n+1)]
    for i in range(1,n+1):
        Q_mat[0][i] = 1/n


    before = QNA_Fluid(n+1,m,ext,c_a,tau,c_s,Q_mat,delta_, TimeInterval)

    n = 2
    #Here, which approximation do we take? average or sum?
    m = [sum(m[1:(len(m))]),100000]

    lam = [ext[0],0]
    delta_ = [1,1]
    t_ = 0
    for i in range(1,len(tau)):
        t_  = max(t_,tau[i])
    tau = [t_,0]
    c_a = [1,0]
    c_s = [1,1]
    Q_mat = [[0,1],[0,0]]
    #m=[1,100000]
    after = QNA_Fluid(n,m,lam,c_a,tau,c_s,Q_mat,delta_, TimeInterval)

    res =(after[0]-before[0])*60
    return res
def CalcSeqCost(tau, m, lam,TimeInterval):
    res = 0
    n = len(tau)
    delta_=[]

    ext = [0 for i in range(0,n)]
    ext[0] = lam


    for i in range(0,n):
        delta_.append(1)

    c_a = [0 for i in range(0,n)]
    c_a[0]=1
    c_s = [1 for i in range(0,n)]
    #c_a.append(0)


    Q_mat = [[0 for i in range(0,n)] for i in range(0,n)]
    for i in range(0,n-1):
        Q_mat[i][i+1] = 1


    before = QNA_Fluid(n,m,ext,c_a,tau,c_s,Q_mat,delta_,TimeInterval)

    n = 2
    #Here, which approximation do we take? average or sum?

    m = [sum(m),100000]

    lam = [ext[0],0]
    delta_ = [1,1]
    t_ = 0
    for i in range(0,len(tau)):
        t_  += tau[i]#/m[i]
    tau = [t_,0]
    c_a = [1,0]
    c_s = [1,1]
    Q_mat = [[0,1],[0,0]]
    #m=[1,100000]
    after = QNA_Fluid(n,m,lam,c_a,tau,c_s,Q_mat,delta_,TimeInterval)

    res =(after[0]-before[0])*60
    return res

#def CalcSeqCost():
def addArrivals(FoldTree, lam, c_sq, state_names):
    for n in FoldTree:
        if n.type =="Activity":
            try:
                name_ind = state_names.index(n.name)
                n.arrival_rate = lam[name_ind]
                n.c_sq = c_sq[name_ind]
            except ValueError:
                print(n.name)

    return FoldTree



def CollectFoldings(FoldTree, Foldings, Types, Precedence, cost, utility,TimeInterval):


    Result=[]
    root_id = 0
    fold_id = 1
    for n in FoldTree:
        if len(n.parents)==0:
            root_id = n.id

    while 1==1:
        if FoldTree[root_id].folded == 1:
            break
        else:
            for n in FoldTree:
                if n.type=="Operator" and n.folded==0:

                    flag = True
                    for c in n.children:
                        if FoldTree[c].folded==0:
                            flag = False
                    if flag == True:
                        if n.name == 'xorLoop':
                        #Hit a leaf - look at the top

                            Foldings.append(fold_id)

                            Types.append('xorLoop')
                            #1. Folded
                            n.folded = 1

                            Prec = n.fold_prev
                            #2. Name
                            for c in n.children:
                                f_id = [FoldTree[c].fold_id]
                                Prec = list(set(f_id).union(set(Prec)))
                                #n.name = n.name+"_"+str(FoldTree[c].name)

                            #3. Service
                            n.service = FoldTree[n.children[0]].service/(1-n.weights)
                            #4. Arrival
                            n.arrival_rate = FoldTree[n.children[0]].arrival_rate
                            #5. c_sq
                            n.c_sq = FoldTree[n.children[0]].c_sq
                            #6. servers
                            n.m = FoldTree[n.children[0]].m
                            #7. type

                            #8. Precedence
                            n.fold_prev = Prec
                            Precedence.append(Prec)
                            cost.append(0)
                            utility.append(1)
                            #9. Weights

                            #10. Children

                            #for c in n.children:
                            #   FoldTree.remove(FoldTree[c])
                            n.fold_id = fold_id
                            fold_id+=1

                        elif n.name=='xor':
                            Foldings.append(fold_id)
                            Types.append('xor')
                            #1. Folded
                            n.folded = 1
                            #2. Type

                            Prec = n.fold_prev
                            #3. Arrival rate
                            #4. c_sq
                            #5. service
                            #6 servers
                            #10. Name
                            n.arrival_rate =0
                            n.c_sq =1
                            n.m = 0
                            tau = []
                            m = []
                            lam_ = []
                            n.service = 0
                            for i,c in enumerate(n.children):
                                f_id = [FoldTree[c].fold_id]
                                Prec = list(set(f_id).union(set(Prec)))
                                n.arrival_rate = n.arrival_rate + FoldTree[c].arrival_rate
                                n.service = n.service+n.weights[i] * FoldTree[c].service #/FoldTree[c].m
                                #n.name = n.name+"_"+str(FoldTree[c].name)
                                n.m +=  FoldTree[c].m
                                tau.append(FoldTree[c].service/3600)
                                m.append(FoldTree[c].m)

                                lam_.append(FoldTree[c].arrival_rate)
                            #n.m = 1

                            #7. Precedence
                            n.fold_prev = Prec
                            Precedence.append(Prec)
                            cost.append(CalcXorCost(n.weights,tau, m, lam_, TimeInterval))
                            #8. weights

                            utility.append(len(n.children))
                            #9. children


                            n.fold_id = fold_id
                            fold_id+=1

                        elif n.name=='and':
                            Foldings.append(fold_id)
                            Types.append('and')
                            #1. Folded
                            n.folded = 1
                            #2. Type

                            Prec = n.fold_prev
                            #3. Arrival rate
                            #4. c_sq
                            #5. service
                            #6 servers
                            #10. Name
                            n.c_sq =1
                            n.m = 0
                            tau = []
                            m = []
                            lam_ = []
                            n.service = 0
                            n.arrival_rate = 0

                            for i,c in enumerate(n.children):
                                f_id = [FoldTree[c].fold_id]
                                Prec = list(set(f_id).union(set(Prec)))
                                n.arrival_rate = max(n.arrival_rate,FoldTree[c].arrival_rate)
                                n.service = max(n.service,FoldTree[c].service) #/FoldTree[c].m)
                                #n.name = n.name+"_"+str(FoldTree[c].name)
                                n.m +=  FoldTree[c].m #/len(n.children)
                                tau.append(FoldTree[c].service/3600)
                                m.append(FoldTree[c].m)
                            lam = n.arrival_rate
                            #n.m = 1
                            #7. Precedence
                            n.fold_prev = Prec
                            Precedence.append(Prec)
                            cost.append(CalcAndCost(tau, m, lam, TimeInterval))
                            #8. weights - no need here.
                            utility.append(len(n.children))
                            #9. children


                            n.fold_id = fold_id
                            fold_id+=1

                        elif n.name=='sequence':
                            Foldings.append(fold_id)
                            Types.append('sequence')
                            #1. Folded
                            n.folded = 1
                            #2. Type

                            Prec = n.fold_prev
                            #3. Arrival rate
                            #4. c_sq
                            #5. service
                            #6 servers
                            #10. Name
                            n.c_sq =1
                            n.m = 0
                            tau = []
                            m = []
                            lam_ = []
                            n.service = 0
                            n.arrival_rate = FoldTree[n.children[0]].arrival_rate

                            for i,c in enumerate(n.children):
                                f_id = [FoldTree[c].fold_id]
                                Prec = list(set(f_id).union(set(Prec)))
                                #n.arrival_rate = max(n.arrival_rate,FoldTree[c].arrival_rate)
                                n.service += FoldTree[c].service #/FoldTree[c].m
                                n.m += FoldTree[c].m
                                #n.name = n.name+"_"+str(FoldTree[c].name)
                                tau.append(FoldTree[c].service/3600)
                                m.append(FoldTree[c].m)
                            lam = n.arrival_rate
                            #7. Precedence
                            n.fold_prev = Prec
                            Precedence.append(Prec)
                            cost.append(CalcSeqCost(tau, m, lam, TimeInterval))
                            #8. weights - no need here.
                            utility.append(len(n.children))
                            #9. children


                            n.fold_id = fold_id
                            fold_id+=1


        #CollectFoldings(FoldTree, Foldings, Types, Precedence,cost,utility)


    return


def getFoldParam(FoldTree,TimeInterval):
    Foldings =[]
    Types = []
    Precedence =[]
    cost = []
    utility = []
    CollectFoldings(FoldTree, Foldings, Types, Precedence,cost,utility,TimeInterval)






    #c,u = CalculateCostsUtilities(Foldings, Types, Annotations)




    res = []
    res.append(Foldings)
    res.append(Precedence)
    res.append(Types)

    res.append(cost)
    res.append(utility)
    return res

def BestBudget(sample, training_results,Bias, train_flag, e_range):

    pred_rmse = []
    pred_mean = sum(sample)/(len(sample)*60)
    total_sq_error = 0

    for s in sample:
        total_sq_error += pow(s/60-pred_mean,2)
    rmse_mean = pow(total_sq_error/len(sample),0.5)

    for prediction in training_results:
      total_sq_error = 0
      for s in sample:
                total_sq_error += pow((Bias+prediction) - s/60,2)
      rmse = pow(total_sq_error/len(sample),0.5)
      pred_rmse.append(rmse)

    mean_pred = [rmse_mean for i in range(0, len(e_range))]
    #pylab.plot(e_range,pred_rmse, 'b-')
    #pylab.ylabel('sRMSE')
    #pylab.xlabel('Budget')
    #pylab.show()


    # if train_flag ==1:
    #     x = e_range
    #     y1 = pred_rmse# Just simulates some data
    #     y2= mean_pred
    #     plt.plot(x, y1, 'b-', label='Training sRMSE', linewidth=4.0)
    #     plt.plot(x, y2, 'b-', label='Training i-sRMSE',linewidth=2.0)
    #
    # else:
    #      font_path = 'C:\Windows\Fonts\Arial.ttf'
    #      font_prop = font_manager.FontProperties(fname=font_path, size=40)
    #      axis_font = {'fontname':'Arial', 'size':'40'}
    #      ax = plt.subplot() # Defines ax variable by creating an empty plot
    #      for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #         label.set_fontname('Arial')
    #         label.set_fontsize(40)
    #      #plt.rcParams.update({'font.size': 26})
    #      x = e_range
    #      y1 = pred_rmse# Just simulates some data
    #      y2= mean_pred
    #      plt.plot(x, y1, 'r--', label='Test sRMSE', linewidth=4.0)
    #      plt.plot(x, y2, 'r--', label='Test i-sRMSE',linewidth=2.0)
    #      plt.xlabel("Budget", **axis_font)
    #      plt.ylabel("sRMSE", **axis_font)
    #      plt.legend(loc='upper right', prop=font_prop, numpoints=1)
    #      pylab.show()

         #pylab.plot(e_range,pred_rmse, 'r--')
         #pylab.plot(e_range,mean_pred, 'r*')
         #pylab.ylabel('sRMSE')
         #pylab.xlabel('Budget')


    min = 1000000
    min_ind = 0
    for i,rm in enumerate(pred_rmse):
            if rm<min:
                min = rm
                min_ind = i

    return min_ind

def RMSETest(sample, best_pred, Bias):
    total_sq_error = 0
    rmse =0
    for s in sample:
      total_sq_error += pow((best_pred+Bias) - s/60,2)
    rmse = pow(total_sq_error/len(sample),0.5)

    return rmse


def testLog(Log, TimeInterval):
    Log = sorted(Log, key = itemgetter(0,2,3))
    cur_id = Log[0][0]
    start = datetime.datetime.strptime(Log[0][2],'%Y/%m/%d %H:%M:%S.%f')
    end = datetime.datetime.strptime(Log[0][3],'%Y/%m/%d %H:%M:%S.%f')
    end_int = datetime.datetime(end.year, end.month,end.day,hour = TimeInterval[1],minute=0,second=0)


    trace=[]
    sample = []
    trace_log = []
    for t in Log:
        if t[0]!=cur_id:
            cur_id = t[0]
            if start<end_int:
                elapsedTime = end-start
                sample.append(divmod(elapsedTime.total_seconds(), 60)[0]*60+divmod(elapsedTime.total_seconds(), 60)[1])
                trace_log.append(trace)
            trace=[]
            trace.append(t[1])
            start = datetime.datetime.strptime(t[2],'%Y/%m/%d %H:%M:%S.%f')
            end = datetime.datetime.strptime(t[3],'%Y/%m/%d %H:%M:%S.%f')
            if end.day>end_int.day:
                    end_int = datetime.datetime(end.year, end.month,end.day,hour = TimeInterval[1],minute=0,second=0)
        else:
            trace.append(t[1])
            cur_end = datetime.datetime.strptime(t[3],'%Y/%m/%d %H:%M:%S.%f')
            if cur_end>end:
                end = cur_end
    print('Sample mean:')
    mean_ = sum(sample)/len(sample)
    print(mean_)
    print('Sample SD:')
    sq_sum = 0
    for s in sample:
        sq_sum+=pow((s-mean_),2)
    print(pow(sq_sum/(len(sample)-1),0.5))

    return sample

best_rmse =[]
#m_range = [i for i in range(8,19)]
m_range = [9]

for k in m_range:
    print(datetime.datetime.now())

    #FoldTree= ReadTreeTuple('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\dfci_april_standard2.ptml')
    print('Reading Tree...')
    #FoldTree = ReadTreeTuple('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\dfci_april_40.ptml')
    FoldTree = ReadTreeTuple('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\dfci_april_75.ptml')
    #FoldTree = ReadTreeTuple('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\dfci_april_50.ptml')
    #FoldTree = ReadTreeTuple('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\dfci_april_65.ptml')

    print('Tree Loaded. Reading Log...')
    Log = getLog('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\DFCI_Train_April.csv')
    #Log = getLog('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\DFCI_April_All_Patients.csv')
    print('Log loaded. Tracifying log...')
    TraceLog = tracifyLog(Log)
    print('Log tracified. Connecting Tree to Log...')
    DayTimeInterval = [k,k+1]
    CTree = ConnectTreeTraceLog(FoldTree,TraceLog)

    ETree = EnrichTree(CTree,Log, TraceLog,DayTimeInterval)
    QNA_Input = UnFoldQNA(ETree)


    n, m,ext,c_ext,tau,c_s,Q_mat,delta_,state_names = QNA_Input
    QNA_FullBlown = QNA_Fluid(n, m,ext,c_ext,tau,c_s,Q_mat,delta_, DayTimeInterval)
    #QNA_FillBlown = QNA(n, m,ext,c_ext,tau,c_s,Q_mat,delta_, DayTimeInterval)
    print('Tree connected and enriched. Adding arrivals...')
    ETree = addArrivals(ETree,QNA_FullBlown[1],QNA_FullBlown[2],state_names)

    param = getFoldParam(ETree,DayTimeInterval)

    #print(QNA_FullBlown[0]*60)
    #res_vars = encode_ILP(c,u,B,S)
    S= []

    for p in param[1]:
        prec =[]
        for k in p:
            if k>0:
                prec.append(k-1)
        S.append(prec)
    c = []
    for p in param[3]:
        c.append(abs(p))
    accel = param[3]

    u = param[4]


    max_B = 0
    for k in c:
        if k>0 and k<1000000:
            max_B+=k
    print ('Maximal budget to consider:')
    print(max_B)
    norm_ = 1

    e_range = [i * norm_ for i in range(0, int((max_B)/norm_))]
    print('Maximal budget:')
    print(e_range[len(e_range)-1])
    # B = 0
    #
    # res_ILP = encode_ILP(c,u,B,S)
    # Foldings_List =[]
    # for r in res_ILP:
    #     Foldings_List.append(r+1)
    # QNA_folded = UnFoldQNANew(ETree,Foldings_List)
    # n, m,ext,c_ext,tau,c_s,Q_mat,delta_,state_names = QNA_folded
    # QNA_Poisson = QNA_Fluid(n, m,ext,c_ext,tau,c_s,Q_mat,delta_,DayTimeInterval)
    # baseline = QNA_Poisson[0]*60


    print('Finding optimal budget...')
    training_results = []
    count_ILP = 0
    for B in e_range:
        count_ILP+=1
        print('CountILP:')
        print(count_ILP)
        delta =0
        res_ILP = encode_ILP(c,u,B,S)
        Foldings_List =[]
        for r in res_ILP:
            Foldings_List.append(r+1)

        #Delta method:
        #for l in Foldings_List:
        #    ind = param[0].index(l)
        #    delta+=accel[ind]
        #training_results.append(baseline + delta)


        #Iterative QNA method:
        QNA_folded = UnFoldQNANew(ETree,Foldings_List)
        n, m,ext,c_ext,tau,c_s,Q_mat,delta_,state_names = QNA_folded
        #= [100000 for i in range(0,n)]
        QNA_Poisson = QNA_Fluid(n, m,ext,c_ext,tau,c_s,Q_mat,delta_,DayTimeInterval)
        training_results.append(QNA_Poisson[0]*60)
        print('Total query time:')
        print(QNA_Poisson[0]*60)






    trainingSample = testLog(Log,DayTimeInterval)
    mean_pred = sum(trainingSample)/(len(trainingSample)*60)
    Bias= mean_pred - training_results[0]
    print('Init Bias:')
    print(Bias)
    Bias = 0
    bb_train = BestBudget(trainingSample, training_results, Bias,1, e_range)
    print('Best Training Budget Is:')
    print(e_range[bb_train])
    rmse = RMSETest(trainingSample,training_results[bb_train], Bias)
    print('Best Training RMSE:')
    print(rmse)

    print('End Bias:')
    print(mean_pred - training_results[bb_train])


    Log = getLog('C:\Users\Arik\PycharmProjects\BPM_ICAPS\Data_Experiments\DFCI_Test_May.csv')
    testSample= testLog(Log,DayTimeInterval)

    bb_test = BestBudget(testSample, training_results, Bias,0, e_range)
    print('Best Test budget:')
    print(e_range[bb_test])

    print('Best Test RMSE:')
    rmse = RMSETest(testSample,training_results[bb_test], Bias)
    print(rmse)

    rmse = RMSETest(testSample,training_results[bb_train], Bias)
    print('Best Train on Test RMSE!!!')
    print(rmse)


    print('Unfolded RMSE!!!')
    unfolded_rmse = RMSETest(testSample,training_results[0], Bias)
    print(unfolded_rmse)

    print('Improvement!!!')
    print(unfolded_rmse-rmse)
    #best_rmse.append(unfolded_rmse-rmse)
    #best_rmse.append(rmse)

    print('Mean RMSE:')
    print(RMSETest(testSample,mean_pred, 0))


    print(datetime.datetime.now())



#
# font_path = 'C:\Windows\Fonts\Arial.ttf'
# font_prop = font_manager.FontProperties(fname=font_path, size=40)
# axis_font = {'fontname':'Arial', 'size':'40'}
# ax = plt.subplot() # Defines ax variable by creating an empty plot
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label.set_fontname('Arial')
#     label.set_fontsize(40)
# #plt.rcParams.update({'font.size': 26})
# x = m_range
# y = best_rmse# Just simulates some data
# plt.plot(x, y, 'r--', label='sRMSE', linewidth=4.0)
# #plt.plot(x, y2, 'r--', label='Test i-sRMSE',linewidth=2.0)
# plt.xlabel("Time-of-Day", **axis_font)
# plt.ylabel("sRMSE", **axis_font)
# #plt.legend(loc='upper right', prop=font_prop, numpoints=1)
# pylab.show()

font_path = 'C:\Windows\Fonts\Arial.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=40)
axis_font = {'fontname':'Arial', 'size':'40'}
ax = plt.subplot() # Defines ax variable by creating an empty plot
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
     label.set_fontname('Arial')
     label.set_fontsize(40)

noise = [15,20,25,30,35,40]
y1 = [166,150,126,140,140,140]
y2= [128,128,128,128,128,128]
plt.plot(noise, y1, 'r--', label='Unfolded Model', linewidth=4.0)
plt.plot(noise, y2, 'b-', label='Best Model', linewidth=4.0)

# #plt.plot(x, y2, 'r--', label='Test i-sRMSE',linewidth=2.0)
plt.xlabel("Noise Filtering Threshold", **axis_font)
plt.ylabel("sRMSE", **axis_font)
plt.legend(loc='upper right', prop=font_prop, numpoints=1)
pylab.show()

#pylab.plot(m_range,best_rmse, 'b')
#pylab.ylabel('Unfolded - Best Folding sRMSE')
#pylab.xlabel('Time-of-Day')
#pylab.show()
#

#Handle bottleneck stations, due to steady state!!!
#QNA_full = QNA(n, m,ext,c_ext,tau,c_s,Q_mat,delta_)
#print(QNA_full[0]*60)

#All Poisson
#
#print(QNA_Poisson[0]*60)
#m_inf = [100000 for i in range(0,n)]
#Infinite Server Prediction:
#QNA_inf = QNA(n, m_inf,ext,c_ext,tau,c_s,Q_mat,delta_)
#print(QNA_inf[0]*60)






#S = [[], [0,2], []]
#c = [20,15,6]
#b = [5,3,3]


#OldTree= deepcopy(ETree)