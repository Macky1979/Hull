import numpy as np

from dill.source import getsource

class Node:
    """
    Michal Mackanic 27/07/2023 v1.0

    This class represents a node on a binomial tree.

    __init__(self, coordinates, S_0, u):
        initiate Node class - initiate a particular node of a binomial tree
        variables:
            coordinates: tuple
                node coordinates in form of (i, j) where i represents time
                step and j total number of upticks (for j > 0) or downticks
                (for j < 0)
            S_0: float
                stock price at time t = 0
            u: float
                percentage increase of stock price on uptick; as we require
                re-combining tree we assume d = 1 / u
    evaluate(df, p, payoff_fnc, opt_tp, f_u, f_d):
        determine derivative price in the node
        variables:
            df: float
                discount factor applicable on the time step
            p: float
                probability of stock price uptick in a risk neutral world
            payoff_fnc: lambda function
                derivative payoff function
            opt_tp: str
                European - European option
                American - American option
            f_u: float
                uptick derivative price from the node in the next time step;
                if f_u is None we assume that the current node is a terminal
                one
            f_d: float
                downtick derivative price from the node in the next time step;
                if f_d is None we assume that the current node is a terminal
                one

    example:
        # node initiation
        coordinates = (1, 1)
        S_0 = 20
        u = 1.1
        node = Node(coordinates=coordinates,
                    S_0=S_0,
                    u=u)

        # node valuation
        df = np.exp(-0.04 * 0.25)
        p = 0.5097
        payoff_fnc = lambda x: max(21.0 - x, 0.0)
        opt_tp = 'European'
        f_u = None
        f_d = None
        node.evaluate(df=df, p=p, payoff_fnc=payoff_fnc, opt_tp=opt_tp, f_u=f_u, f_d=f_d)
    """
    def __init__(self, coordinates, S_0, u):

        # store variables into the object
        self.coordinates = coordinates

        # perform node specific checks
        if self.coordinates[0] < 0:
            raise ValueError ('Node ' + str(self.coordinates) + ': ' +\
                              'the first coordinate represents number of ' +\
                              'time steps and as such cannot be negative!')

        if self.coordinates[1] > self.coordinates[0]:
            raise ValueError ('Node ' + str(self.coordinates) + ': ' +\
                              'the second  coordinate cannot be higher ' +\
                              'than the first coordinate!')

        # determine stock price at the node
        if self.coordinates[1] > 0:
            self.S = S_0 * (u ** self.coordinates[1])
        elif self.coordinates[1] == 0:
            self.S = S_0
        else:
            self.S = S_0 * ((1 / u) ** -self.coordinates[1])

        # early exercise indicator for American option
        self.exe = False

    def evaluate(self, df, p, payoff_fnc, opt_tp, f_u=None, f_d=None):

        # check that either both or none of f_u and f_d were specified
        if (((f_u is None) and (f_d is not None)) or
            ((f_u is not None) and (f_d is None))):

            raise ValueError ('Node ' + str(self.coordinates) + ': ' +\
                              'either both or none of f_u and f_d ' +\
                              'are specified!')

        # if (a) f_u and f_d are not specified, the node belongs to terminal
        # nodes or (b) it is an American option => we have to evaluate
        # derivative payoff function
        if (f_u is None) or (opt_tp == 'American'):
            self.f = payoff_fnc(self.S)

        # European option and non-terminal node
        if (f_u is not None) and (opt_tp == 'European'):
            self.f = (p * f_u + (1 - p) * f_d) * df

        # American option and non-terminal node
        if (f_u is not None) and (opt_tp == 'American'):
            f_aux = (p * f_u + (1 - p) * f_d) * df

            if self.f > f_aux:
                self.exe = True
            else:
                self.exe = False
                self.f = f_aux


class Tree:
    """
    Michal Mackanic 28/07/2023 v1.0

    This class represents a binomial tree. For theoretical background see
    chapter 13 in "Optiones, Futures, and Other Derivates", 11th edition by
    John Hull

    __init__(self, S_0, sigma, r, q, payoff_fnc, T, time_steps_no=100):
        initiate binomial tree class
        variables:
            S_0: float
                stock price at time t = 0
            sigma: float
                annualized standard deviation of stock price
            r: float
                risk-free rate
            q: float
                continuous dividend yield
            payoff_fnc: lambda function
                derivative payoff function
            T: float
                residual maturity in years
            time_steps_no: int
                number of time steps within the binomial tree
    get_node(coordinates):
        get node of the tree
        variables:
            coordinates: tuple
                node coordinates in form of (i, j) where i represents time
                step and j total number of upticks (for j > 0) or downticks
                (for j < 0)
    evaluate(opt_tp):
        evaluate binomial tree
        opt_tp: str
            European - European option
            American - American option
    get_value():
        get option value stored in top node defined through (0, 0) coordinates
    print(file_nm, time_step_len):
        store all information about the tree and its individual nodes into
        a text file
        variables:
            file_nm: str
                name of file to which the tree related information is supposed
                to be stored
            time_step_len: int
                "width" of individual nodes in the file

    example 1:
        # figure 13.11
        S_0 = 810
        sigma = 0.20
        r = 0.05
        q = 0.02
        payoff_fnc = lambda x: max(x - 800, 0)
        T = 0.50
        time_steps_no = 2
        opt_tp='European'
        tree = Tree(S_0=S_0,
                    sigma=sigma,
                    r=r,
                    q=q,
                    payoff_fnc=payoff_fnc,
                    T=T,
                    time_steps_no=time_steps_no)
        tree.evaluate(opt_tp=opt_tp)
        tree.print('tree_1.txt')
        print(tree.get_value())

    example 2:
        # figure 13.12
        S_0 = 0.610
        sigma = 0.12
        r = 0.05
        q = 0.07
        payoff_fnc = lambda x: max(x - 0.600, 0)
        T = 0.25
        time_steps_no = 3
        opt_tp='American'
        tree = Tree(S_0=S_0,
                    sigma=sigma,
                    r=r,
                    q=q,
                    payoff_fnc=payoff_fnc,
                    T=T,
                    time_steps_no=time_steps_no)
        tree.evaluate(opt_tp=opt_tp)
        tree.print('tree_2.txt')
        print(tree.get_value())

    example 3:
        # figure 13.13
        S_0 = 31
        sigma = 0.30
        r = 0.05
        q = 0.05  # futures contract earns risk-free rate by definition
        payoff_fnc = lambda x: max(30 - x, 0)
        T = 0.75
        time_steps_no = 3
        opt_tp='American'
        tree = Tree(S_0=S_0,
                    sigma=sigma,
                    r=r,
                    q=q,
                    payoff_fnc=payoff_fnc,
                    T=T,
                    time_steps_no=time_steps_no)
        tree.evaluate(opt_tp=opt_tp)
        tree.print('tree_3.txt')
        print(tree.get_value())
    """

    def __init__(self,
                 S_0,
                 sigma,
                 r,
                 q,
                 payoff_fnc,
                 T,
                 time_steps_no=100,
                 opt_tp='European'):

        # store variables into the object
        self.S_0 = S_0
        self.sigma = sigma
        self.r = r
        self.q = q
        self.payoff_fnc = payoff_fnc
        self.T = T
        self.time_steps_no = time_steps_no
        self.opt_tp=opt_tp

        # get time step length
        self.dt = self.T / self.time_steps_no

        # get discount factor per time step assuming fixed time step length
        # and constant risk-free rate
        self.df = np.exp(-self.r * self.dt)

        # determine relative price change of stock during uptick
        self.u = np.exp(self.sigma * np.sqrt(self.dt))

        # relative price change of stock during downtick is defined as d = 1 / u
        # so that the binomial tree is recombining
        self.d = 1 / self.u

        # determine probability of uptick in risk-neutral world
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

        # initiate nodes of the binomial tree
        self.nodes = {}
        for i in range(self.time_steps_no + 1):

            for j in range(-i, i + 1, 2):

                coordinates = (i, j)
                self.nodes[coordinates] = Node(coordinates=coordinates,
                                               S_0=self.S_0,
                                               u=self.u)

    def get_node(self, coordinates):

        # return node
        try:
            return self.nodes[coordinates]
        except KeyError:
            raise KeyError('Node ' + str(coordinates) + ' not found!')


    def evaluate(self, opt_tp='European'):

        # store option type
        self.opt_tp = opt_tp

        # go backwards through time steps
        for i in range(self.time_steps_no, -1, -1):

            # go through individual nodes on a given time step
            for j in range(-i, i + 1, 2):

                # teminal nodes
                if i == self.time_steps_no:
                    f_u = None
                    f_d = None

                # other nodes
                else:

                    coordinates = (i + 1, j + 1)
                    f_u = self.get_node(coordinates).f

                    coordinates = (i + 1, j - 1)
                    f_d = self.get_node(coordinates).f

                # evaluate node
                coordinates = (i, j)
                self.get_node(coordinates).evaluate(df=self.df,
                                                    p=self.p,
                                                    payoff_fnc=self.payoff_fnc,
                                                    opt_tp=self.opt_tp,
                                                    f_u=f_u,
                                                    f_d=f_d)

    def get_value(self):
        coordinates = (0, 0)
        return self.get_node(coordinates).f

    def print(self, file_nm, time_step_len=9):

        # open file
        f = open(file_nm, 'w')

        # print parameters
        f.write('TREE DEFINITION' + '\n')
        f.write('option type: ' + self.opt_tp + '\n')
        f.write(getsource(self.payoff_fnc))
        f.write('S_0 = ' + '{:,.2f}'.format(self.S_0) + '\n')
        f.write('sigma = ' + '{:,.2%}'.format(self.sigma) + '\n')
        f.write('r = ' + '{:,.2%}'.format(self.r) + '\n')
        f.write('q = ' + '{:,.2%}'.format(self.q) + '\n')
        f.write('T = ' + '{:,.2f}'.format(self.T) + '\n')
        f.write('time_steps_no = ' + '{:,.0f}'.format(self.time_steps_no) + '\n')
        f.write('\n')
        f.write('DERIVED PARAMETERS' + '\n')
        f.write('dt = ' + '{:,.2f}'.format(self.dt) + '\n')
        f.write('df = ' + '{:,.2%}'.format(self.df) + '\n')
        f.write('p = ' + '{:,.2%}'.format(self.p) + '\n')
        f.write('u = ' + '{:,.5f}'.format(self.u) + '\n')
        f.write('d = ' + '{:,.5f}'.format(self.u) + '\n')
        f.write('\n')

        # print the tree
        f.write('BINOMIAL TREE' + '\n')

        f.write('\n')
        f.write('node coordinates' + '\n')
        f.write('stock price' + '\n')
        f.write('derivative price' + '\n')
        f.write('early exercise indicator' + '\n')
        f.write('\n')

        for j in range(self.time_steps_no, -self.time_steps_no - 1, -1):

            row_1 = ''
            row_2 = ''
            row_3 = ''
            row_4 = ''

            for i in range(self.time_steps_no + 1):

                # try to get a node
                try:

                    # get coordinates
                    coordinates = (i, j)
                    node = self.get_node(coordinates)

                    coordinates = str(coordinates)
                    adj = time_step_len - len(coordinates)
                    coordinates += ' ' * adj
                    row_1 += coordinates

                    # get stock price at the node
                    S = '{:,.3f}'.format(node.S)
                    adj = time_step_len - len(S)
                    S += ' ' * adj
                    row_2 += S

                    # get derivative price
                    val = '{:,.3f}'.format(node.f)
                    adj = time_step_len - len(val)
                    val += ' ' * adj
                    row_3 += val

                    # indicator of early exercise in case of American option
                    if self.opt_tp == 'American':
                        exe_ic = str(node.exe)
                        adj = time_step_len - len(exe_ic)
                        exe_ic += ' ' * adj
                        row_4 += exe_ic

                # no node found
                except KeyError:

                    row_1 += ' ' * time_step_len
                    row_2 += ' ' * time_step_len
                    row_3 += ' ' * time_step_len
                    if self.opt_tp == 'American':
                        row_4 += ' ' * time_step_len

            # add rows
            f.write(row_1 + '\n')
            f.write(row_2 + '\n')
            f.write(row_3 + '\n')
            if self.opt_tp == 'American':
                f.write(row_4 + '\n')

            # next node level
            f.write('\n')

        # close file
        f.close()
