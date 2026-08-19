"""Rebuild the canonical notebook as Markdown plus one-line presentation helper calls."""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK = Path("notebooks/gwp2_vix_regime_allocation.ipynb")


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def call(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def main() -> None:
    cells = [
        md(
            r"""
# MScFE 622 Stochastic Modeling — Group Work Project #2
## VIX-Regime Allocation across TLT, GLD, and SPY

This is the canonical technical report notebook. It deliberately contains **no analysis implementation**: calculations, validation, file loading, formatting, and figure presentation live in tested helper modules under `src/vix_regime_allocation/`. Notebook code cells are restricted to one import and one-line helper calls so that the mathematical reasoning and empirical interpretation remain visually separate from implementation details.

The empirical design uses daily Yahoo Finance adjusted closes over the maximum common sample, daily ETF log returns, and the daily first difference of VIX. It compares two- and three-state quantile Markov specifications with two- and three-state Gaussian hidden-state specifications, selects the preferred regime representation using the pre-declared decision rule, maps each selected state to the ETF with the largest historical conditional mean return, and evaluates the resulting one-row-lagged allocation against monthly equal weight and SPY buy-and-hold benchmarks.
"""
        ),
        call("import vix_regime_allocation.notebook_helpers as nb"),
        md(
            r"""
## 1. Data preparation and exploratory analysis

The market sample contains TLT, GLD, SPY, and VIX only on dates for which all four adjusted-close observations are present. No forward fill, backward fill, or interpolation is used. Adjusted closes are appropriate for the ETF return series because they incorporate distribution and split adjustments supplied by the data source; the VIX series is treated as the published index level rather than as a tradable asset price.

The VIX is a forward-looking option-implied measure of expected S&P 500 volatility over approximately the next 30 days, not a forecast of the direction of the equity index (Whaley 98–105; Cboe Global Markets).

<!-- citekey: whaley2009vix -->
<!-- citekey: cboe2019vixfaq -->

### ETF log returns
For ETF \(i\) on observed trading row \(t\), the daily log return is

\[
r_{i,t}=\ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
\]

Here \(P_{i,t}\) is the adjusted close and \(r_{i,t}\) is the continuously compounded one-period return. Log returns are additive across time, while portfolio arithmetic in the backtest later uses simple returns obtained by \(R_{i,t}=e^{r_{i,t}}-1\).

### VIX change
**Greek letter used below:** \(\Delta\) — *Delta* — denotes a first difference.

\[
\Delta VIX_t = VIX_t-VIX_{t-1}.
\]

Using the change rather than the level makes the state variable represent the direction and magnitude of daily volatility shocks: negative values indicate falling implied volatility and positive values indicate rising implied volatility.
"""
        ),
        call("nb.show_step1_sample()"),
        call("nb.show_step1_figures()"),
        md(
            r"""
### Interpretation of the exploratory figures

The ETF return series fluctuate around zero and exhibit volatility clustering and occasional large observations. VIX changes are much more visibly heavy-tailed: ordinary days are concentrated near zero, while crisis periods generate large positive jumps and subsequent reversals. This asymmetry is economically important because an allocation rule based on VIX changes is designed to react to changes in perceived market stress rather than to the absolute VIX level alone.
"""
        ),
        md(
            r"""
## 2. Regime modeling
### 2.1 Quantile-discretized Markov chain

For each candidate state count \(K\in\{2,3\}\), the empirical distribution of daily VIX changes is partitioned by sample quantiles. With \(K=2\), the median separates the two states. With \(K=3\), the one-third and two-thirds quantiles produce three ordered states. An observation exactly equal to a cut point is assigned to the higher state, which makes the state-definition rule deterministic.

If \(N_{ij}\) is the number of observed transitions from state \(i\) to state \(j\), the maximum-likelihood transition estimate is

\[
\widehat P_{ij}=\frac{N_{ij}}{\sum_{m=0}^{K-1}N_{im}}.
\]

**Greek letter used below:** \(\pi\) — *pi* — denotes the stationary state-probability row vector.

A stationary distribution satisfies

\[
\boldsymbol{\pi}\widehat P=\boldsymbol{\pi},
\qquad
\sum_{j=0}^{K-1}\pi_j=1.
\]

The stationary vector is a long-run probability description implied by the estimated transition matrix; it is not a forecast that the market will remain permanently in those proportions.
"""
        ),
        call("nb.show_step2_markov_results()"),
        md(
            r"""
### 2.2 Gaussian hidden-state specification

The alternative model treats the observed daily VIX change as generated by an unobserved state process. Conditional on state \(S_t=j\), each observation follows a state-specific Gaussian distribution.

**Greek letters used below:** \(\mu\) — *mu* — is a state-conditional mean; \(\sigma\) — *sigma* — is a state-conditional standard deviation.

\[
X_t\mid S_t=j\sim\mathcal N(\mu_j,\sigma_j^2),
\qquad X_t=\Delta VIX_t.
\]

The hidden state follows a first-order Markov transition law,

\[
P(S_t=j\mid S_{t-1}=i)=p_{ij}.
\]

Parameters are estimated by likelihood maximization with deterministic multiple restarts. The fitted states are relabeled by increasing state-conditional mean VIX change, so state numbering has an economically consistent interpretation across candidate fits. The Viterbi sequence is the single most likely state path under the fitted model, whereas the smoothed posterior probabilities quantify uncertainty in state membership after conditioning on the full observed sequence (Baum et al. 164–171; Rabiner 257–286; Viterbi 260–269).

<!-- citekey: baum1970maximization -->
<!-- citekey: rabiner1989tutorial -->
<!-- citekey: viterbi1967decoding -->
"""
        ),
        call("nb.show_step2_hmm_results()"),
        md(
            r"""
## 3. Model comparison and economic state interpretation

For a candidate with maximized log-likelihood \(\ell\), parameter count \(k\), and observation count \(n\), the information criteria are

\[
AIC=2k-2\ell,
\qquad
BIC=k\ln(n)-2\ell.
\]

Lower values indicate a better likelihood-versus-complexity trade-off within a comparable likelihood family (Akaike 716–723; Schwarz 461–464). In this project, the quantile Markov likelihood is defined on a discretized transition sequence while the hidden-state likelihood is defined on the continuous VIX-change observations. Therefore the raw AIC/BIC magnitudes are **not** interpreted as a statistically valid cross-family ranking. BIC selects the preferred state count within each family; the pre-declared diagnostic rule then determines which family supplies the allocation states.

<!-- citekey: akaike1974identification -->
<!-- citekey: schwarz1978dimension -->
"""
        ),
        call("nb.show_step3_model_selection()"),
        md(
            r"""
### State-conditional ETF moments

For state \(s\), asset \(i\), and \(n_s\) observations, the conditional daily mean log return is

\[
\bar r_{i,s}=\frac{1}{n_s}\sum_{t:S_t=s} r_{i,t},
\]

and the sample standard deviation is

\[
s_{i,s}=\sqrt{\frac{1}{n_s-1}\sum_{t:S_t=s}\left(r_{i,t}-\bar r_{i,s}\right)^2}.
\]

The plotted error bars are **sample standard deviations**, not standard errors or confidence intervals. They therefore describe the dispersion of daily returns within a state rather than statistical uncertainty around the estimated conditional mean.
"""
        ),
        call("nb.show_step3_state_statistics()"),
        md(
            r"""
## 4. State-based allocation rule

For every preferred state \(s\), the allocation assigns 100% of capital to the ETF with the largest historical state-conditional mean log return:

\[
a^*(s)=\operatorname*{arg\,max}_{a\in\{TLT,GLD,SPY\}}\bar r_{a,s}.
\]

The corresponding portfolio weight is

\[
w_a(s)=\mathbf 1\{a=a^*(s)\}.
\]

Exact ties are resolved by the fixed priority TLT, then GLD, then SPY. This deterministic rule makes the mapping reproducible, but it is deliberately concentrated rather than diversified. Markowitz's portfolio framework reminds us that maximizing a conditional mean alone does not control covariance or total portfolio risk (Markowitz 77–91).

<!-- citekey: markowitz1952portfolio -->
"""
        ),
        call("nb.show_step4_allocation()"),
        md(
            r"""
## 5. Backtest and evaluation
### One-observed-row execution lag

The state observed on trading row \(t-1\) determines the weights applied to the ETF returns on row \(t\). ETF log returns are first converted to simple returns before portfolio aggregation:

\[
R_{i,t}=e^{r_{i,t}}-1,
\qquad
R^{rot}_t=\sum_i w_i(S_{t-1})R_{i,t}.
\]

This lag prevents using the same row's state to select the same row's return. It does **not**, by itself, create an out-of-sample experiment because the regime definitions and state-conditional allocation means were estimated from the full sample.

### Benchmarks
The comparison uses exactly two benchmarks on the same return dates: (1) one-third TLT, one-third GLD, and one-third SPY, reset to equal weights at the first observed trading date of each calendar month and allowed to drift intra-month; and (2) SPY buy-and-hold.

### Performance measures
With initial wealth \(W_0=1\) and daily simple portfolio return \(R_t\),

\[
W_T=\prod_{t=1}^{T}(1+R_t),
\qquad
CR=W_T-1.
\]

Annualized return is

\[
AR=W_T^{252/T}-1,
\]

annualized volatility is

\[
AV=s_R\sqrt{252},
\]

and the zero-risk-free Sharpe ratio is

\[
SR=\frac{\bar R}{s_R}\sqrt{252}.
\]

Maximum drawdown uses the running wealth peak including \(W_0=1\):

\[
MDD=\min_t\left(\frac{W_t}{\max_{0\le u\le t}W_u}-1\right).
\]

Including initial wealth is necessary so that an immediate first-period loss is correctly recognized as a drawdown.
"""
        ),
        call("nb.show_step5_backtest()"),
        md(
            r"""
### State-count sensitivity

The final sensitivity check compares \(K=2\) and \(K=3\) **within the already selected model family** on a common lagged-return date intersection. It is a robustness description, not a second model-selection exercise, and it must not replace the Step 3 selection rule merely because one historical backtest row performs better.
"""
        ),
        call("nb.show_step5_sensitivity()"),
        md(
            r"""
## 6. Interpretation, limitations, and practitioner implications

The project demonstrates that VIX-change states can be converted into a transparent allocation policy, but the historical backtest must be interpreted cautiously. The full-sample regime thresholds or fitted parameters, the full-sample state sequence, and the full-sample state-conditional ETF means all use information unavailable at earlier decision dates. The required one-row trading lag corrects same-row execution timing but does not remove that estimation look-ahead.

A stronger production study would use rolling or expanding estimation, one-sided state inference, allocation means estimated only from information available at each decision date, explicit transaction costs and turnover, and repeated robustness checks across samples. White emphasizes the danger of data-snooping inference, while Bailey and López de Prado show that conventional Sharpe-ratio interpretation can be badly distorted by selection bias and backtest overfitting (White 1097–1126; Bailey and López de Prado 94–107).

<!-- citekey: white2000datasnooping -->
<!-- citekey: bailey2014deflatedsharpe -->

From a Financial Engineering perspective, the key contribution is the separation of four distinct layers: state inference, conditional return estimation, deterministic allocation, and performance evaluation. That separation makes it possible to replace any single layer—such as the regime model or allocation objective—without changing the rest of the research pipeline.
"""
        ),
        md(
            r"""
## Works Cited

Akaike, Hirotugu. “A New Look at the Statistical Model Identification.” *IEEE Transactions on Automatic Control*, vol. 19, no. 6, 1974, pp. 716–723. doi:10.1109/TAC.1974.1100705.

Bailey, David H., and Marcos López de Prado. “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.” *The Journal of Portfolio Management*, vol. 40, no. 5, 2014, pp. 94–107. doi:10.3905/jpm.2014.40.5.094.

Baum, Leonard E., et al. “A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains.” *The Annals of Mathematical Statistics*, vol. 41, no. 1, 1970, pp. 164–171. doi:10.1214/aoms/1177697196.

Cboe Global Markets. “Cboe VIX FAQ.” *Cboe Global Markets*, 2019. Accessed 19 Aug. 2026.

Markowitz, Harry. “Portfolio Selection.” *The Journal of Finance*, vol. 7, no. 1, 1952, pp. 77–91. doi:10.1111/j.1540-6261.1952.tb01525.x.

Rabiner, Lawrence R. “A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.” *Proceedings of the IEEE*, vol. 77, no. 2, 1989, pp. 257–286. doi:10.1109/5.18626.

Schwarz, Gideon. “Estimating the Dimension of a Model.” *The Annals of Statistics*, vol. 6, no. 2, 1978, pp. 461–464. doi:10.1214/aos/1176344136.

Viterbi, Andrew J. “Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm.” *IEEE Transactions on Information Theory*, vol. 13, no. 2, 1967, pp. 260–269. doi:10.1109/TIT.1967.1054010.

Whaley, Robert E. “Understanding the VIX.” *The Journal of Portfolio Management*, vol. 35, no. 3, 2009, pp. 98–105. doi:10.3905/JPM.2009.35.3.098.

White, Halbert. “A Reality Check for Data Snooping.” *Econometrica*, vol. 68, no. 5, 2000, pp. 1097–1126. doi:10.1111/1468-0262.00152.
"""
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"rebuilt {NOTEBOOK} with {len(cells)} cells")


if __name__ == "__main__":
    main()
