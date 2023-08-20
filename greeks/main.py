import numpy as np
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader

# import Black-Scholes formulas
file_nm = '/home/macky/Documents/Hull/black_scholes/main.py'
bs = SourceFileLoader('main', file_nm).load_module()


def main(K, r, q, sigma, **kwargs):
    """
    Michal Mackanic 20/08/2023 v1.0

    This function plots greeks in line with chapter 19 of "Options, Futures,
    and Other Derivatives", e11 by John Hull. If parameter T is specified,
    greeks are ploted vs. range of spot prices of the underlying stock. If
    parameter S_0 is specified, greeks are ploted vs. range of option
    maturity.

    input:
        K: float
            strike
        r: float
            continuous annual risk-free rate
        sigma: float
            annual standard deviation entering Black-Scholes formula
        T: float (either variable T or S_0 - see example 1 & 2)
            option maturity in years
        S_0: float (either variable T or S_0 - see example 1 & 2)
            spot price of the underlying stock
    output:
        plots stored in folder greeks_vs_maturity (if parameter S_0 is
        specified) or greeks_vs_spot_price (if parameter T is specified)

    example 1:
        # greeks vs. spot price of the underlying stock
        K = 50
        r = 0.05
        q = 0.03
        sigma = 0.20
        T = 2.0
        main(K=K, r=r, q=q, sigma=sigma, T=T)

    example 2:
        # greeks vs. option maturity
        K = 50
        r = 0.05
        q = 0.03
        sigma = 0.20
        S_0 = 50
        main(K=K, r=r, q=q, sigma=sigma, S_0=S_0)
    """

    # determine if we will interate through spot price of maturity
    for key, value in kwargs.items():

        # lists to hold greeks
        delta_call = []
        delta_put = []
        gamma_call = []
        gamma_put = []
        theta_call = []
        theta_put = []
        vega_call = []
        vega_put = []
        rho_r_call = []
        rho_r_put = []
        rho_q_call = []
        rho_q_put = []

        #### iterate spot price
        if (key == 'T'):

            # inform user
            print('Greeks vs. spot price')

            # get maturity value
            T = value

            # get stock prices
            S_0_list = np.linspace(1, 3 * K, 100)

            #### calculating greeks
            print('   Calculating greeks ...')

            # go through individual spot prices
            for S_0 in S_0_list:

                # calculate greeks for call option for a particular spot price
                call = bs.BlackScholes(tp='call',
                                      S_0=S_0,
                                      K=K,
                                      r=r,
                                      q=q,
                                      sigma=sigma,
                                      T=T,
                                      greeks=True)

                # calculate greeks for put option for a particular spot price
                put = bs.BlackScholes(tp='put',
                                      S_0=S_0,
                                      K=K,
                                      r=r,
                                      q=q,
                                      sigma=sigma,
                                      T=T,
                                      greeks=True)

                # store deltas
                delta_call.append(call.greeks['delta'])
                delta_put.append(put.greeks['delta'])

                # store gammas
                gamma_call.append(call.greeks['gamma'])
                gamma_put.append(put.greeks['gamma'])

                # store thetas
                theta_call.append(call.greeks['theta'])
                theta_put.append(put.greeks['theta'])

                # store vegas
                vega_call.append(call.greeks['vega'])
                vega_put.append(put.greeks['vega'])

                # store rho_r
                rho_r_call.append(call.greeks['rho_r'])
                rho_r_put.append(put.greeks['rho_r'])

                # store rho_q
                rho_q_call.append(call.greeks['rho_q'])
                rho_q_put.append(put.greeks['rho_q'])

            #### plotting greeks
            print('   Plotting greeks ...')

            # target folder
            target_folder_nm = 'greeks_vs_spot_price'

            # plot deltas
            plot_greek(x=S_0_list,
                       ys=[delta_call, delta_put],
                       x_label='$S_0$',
                       plot_nm= 'Delta',
                       file_nm=target_folder_nm + '//delta_vs_spot_price.png')

            # plot gammas
            plot_greek(x=S_0_list,
                       ys=[gamma_call, gamma_put],
                       x_label='$S_0$',
                       plot_nm= 'Gamma',
                       file_nm=target_folder_nm + '//gamma_vs_spot_price.png')

            # plot thetas
            plot_greek(x=S_0_list,
                       ys=[theta_call, theta_put],
                       x_label='$S_0$',
                       plot_nm= 'Theta',
                       file_nm=target_folder_nm + '//theta_vs_spot_price.png')

            # plot vegas
            plot_greek(x=S_0_list,
                       ys=[vega_call, vega_put],
                       x_label='$S_0$',
                       plot_nm= 'Vega',
                       file_nm=target_folder_nm + '//vega_vs_spot_price.png')

            # plot rho_r
            plot_greek(x=S_0_list,
                       ys=[rho_r_call, rho_r_put],
                       x_label='$S_0$',
                       plot_nm= 'Rho of r',
                       file_nm=target_folder_nm + '//rho_r_vs_spot_price.png')

            # plot rho_q
            plot_greek(x=S_0_list,
                       ys=[rho_q_call, rho_q_put],
                       x_label='$S_0$',
                       plot_nm= 'Rho of q',
                       file_nm=target_folder_nm + '//rho_q_vs_spot_price.png')

            # inform user about saved plots
            print('   Plots with greeks have been saved in folder ' + target_folder_nm + '!')

        #### iterate through maturity
        elif (key == 'S_0'):

            # inform user
            print('Greeks vs. maturity')

            # get spot price
            S_0 = value

            # get maturities
            T_list = np.linspace(1, 10, 10)

            #### calculating greeks
            print('   Calculating greeks ...')

            # go through individual maturities
            for T in T_list:

                # calculate greeks for call option for a particular maturity
                call = bs.BlackScholes(tp='call',
                                      S_0=S_0,
                                      K=K,
                                      r=r,
                                      q=q,
                                      sigma=sigma,
                                      T=T,
                                      greeks=True)

                # calculate greeks for put option for a particular maturity
                put = bs.BlackScholes(tp='put',
                                       S_0=S_0,
                                       K=K,
                                       r=r,
                                       q=q,
                                       sigma=sigma,
                                       T=T,
                                       greeks=True)

                # store deltas
                delta_call.append(call.greeks['delta'])
                delta_put.append(put.greeks['delta'])

                # store gammas
                gamma_call.append(call.greeks['gamma'])
                gamma_put.append(put.greeks['gamma'])

                # store thetas
                theta_call.append(call.greeks['theta'])
                theta_put.append(put.greeks['theta'])

                # store vegas
                vega_call.append(call.greeks['vega'])
                vega_put.append(put.greeks['vega'])

                # store rho_r
                rho_r_call.append(call.greeks['rho_r'])
                rho_r_put.append(put.greeks['rho_r'])

                # store rho_q
                rho_q_call.append(call.greeks['rho_q'])
                rho_q_put.append(put.greeks['rho_q'])

            ##### plotting greeks
            print('   Plotting greeks ...')

            # target folder
            target_folder_nm = 'greeks_vs_maturity'

            # plot deltas
            plot_greek(x=T_list,
                       ys=[delta_call, delta_put],
                       x_label='maturity',
                       plot_nm= 'Delta',
                       file_nm=target_folder_nm + '//delta_vs_maturity.png')

            # plot gammas
            plot_greek(x=T_list,
                       ys=[gamma_call, gamma_put],
                       x_label='maturity',
                       plot_nm= 'Gamma',
                       file_nm=target_folder_nm + '//gamma_vs_maturity.png')

            # plot thetas
            plot_greek(x=T_list,
                       ys=[theta_call, theta_put],
                       x_label='maturity',
                       plot_nm= 'Theta',
                       file_nm=target_folder_nm + '//theta_vs_maturity.png')

            # plot vegas
            plot_greek(x=T_list,
                       ys=[vega_call, vega_put],
                       x_label='maturity',
                       plot_nm= 'Vega',
                       file_nm=target_folder_nm + '//vega_vs_maturity.png')

            # plot rho_r
            plot_greek(x=T_list,
                       ys=[rho_r_call, rho_r_put],
                       x_label='maturity',
                       plot_nm= 'Rho of r',
                       file_nm=target_folder_nm + '//rho_r_vs_maturity.png')

            # plot rho_q
            plot_greek(x=T_list,
                       ys=[rho_q_call, rho_q_put],
                       x_label='maturity',
                       plot_nm= 'Rho of q',
                       file_nm=target_folder_nm + '//rho_q_vs_maturity.png')

            # inform user about saved plots
            print('   Plots with greeks have been saved in folder ' + target_folder_nm + '!')

        # unsupported parameter
        else:
            raise ValueError('Only additional parameters that are supported ' +\
                             ' are T and S_0!')


def plot_greek(x, ys, x_label, plot_nm, file_nm):
    """
    Michal Mackanic 20/08/2023 v1.0

    This an auxiliary function called by main(), which plots greeks.

    input:
        x: list of floats
            x-axis values; list of spot prices (if parameter T is specified)
            or list of maturities (if parameter S_0 is specified)
        ys: list of two lists of floats
            list of greeks for call and put option
        x_label: str
            label of x-axis
        plot_nm: str
            plot title
        file_nm: str
            file name under which the plot should be stored
    output:
        plot stored in folder greeks_vs_maturity (if parameter S_0 is
        specified) or greeks_vs_spot_price (if parameter T is specified)

    example:
        see function main()
    """

    # clear graph
    plt.clf()

    # greek of call option
    fig, ax1 = plt.subplots()
    plt1 = ax1.plot(x,
                    ys[0],
                    linestyle='solid',
                    linewidth=1.0,
                    color='red',
                    label='call')

    # greek of put option
    ax2 = ax1.twinx()
    plt2 = ax2.plot(x,
                    ys[1],
                    linestyle='solid',
                    linewidth=1.0,
                    color='blue',
                    label='put')

    # title
    plt.title(plot_nm)

    # add legend
    plts = plt1 + plt2
    labels = [plt_.get_label() for plt_ in plts]
    plt.legend(plts, labels, loc=0)

    # label axis
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('call option value')
    ax2.set_ylabel('put option value')

    # save plot
    plt.tight_layout()
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(file_nm, dpi=100)