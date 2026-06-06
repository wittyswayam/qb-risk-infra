# qb-risk-infra

A quantitative backtesting and risk analytics infrastructure built in Python. The system provides modular, independently testable components for strategy evaluation, walk-forward validation, Monte Carlo stress simulation, and portfolio risk analytics. It is designed to be extended, not just run.

---

## Overview

This project exists because most publicly available backtesting tools conflate signal generation with execution modelling, omit transaction costs, or provide no path from in-sample results to out-of-sample validation. The goal here is a platform where each concern — data ingestion, strategy logic, order routing, portfolio accounting, risk measurement — is handled by a distinct, testable module.

The system is built to be used analytically. It is not a live trading engine. There are no claims of alpha generation, no ML-based price prediction, and no hidden performance statistics. What it does provide is a rigorous framework for asking structured questions: how does a strategy behave under realistic cost assumptions, how stable are its parameters across time, and how does its portfolio distribution look under stressed conditions?

---

## Architecture

The codebase is organised around a layered architecture with strict dependency flow:

```
Ingestion Layer
    └── Data validation → normalised OHLCV DataFrames

Strategy Layer
    └── BaseStrategy interface → Signal objects

Backtesting Engine
    └── OrderRouter → Portfolio → PortfolioSnapshot → BacktestResult

Validation Layer
    └── WalkForwardValidator → WalkForwardResult

Simulation Layer
    └── MonteCarloSimulator → MonteCarloResult

Risk Analytics
    └── Rolling vol, VaR, CVaR, beta, correlation

Explainability Layer
    └── FeatureBuilder → SHAPAnalyzer → InterpretabilityReport

Persistence Layer
    └── SQLAlchemy models → Repository classes → PostgreSQL

API Layer
    └── FastAPI routes → Pydantic schemas → domain modules

Dashboard
    └── Streamlit app → direct module imports
```

The domain modules (`strategies`, `backtesting`, `montecarlo`, `risk`) have no knowledge of FastAPI or SQLAlchemy. The API and persistence layers are adapters that wrap the domain. This separation makes it practical to write unit tests against the domain without mocking HTTP or database infrastructure.

### Key engineering decisions

**Event-driven bar iteration.** The backtesting engine iterates bar-by-bar rather than computing signals on the entire series at once. This eliminates the most common form of look-ahead bias, where future prices influence signal generation or position sizing. The strategy receives only history up to and including the current bar.

**Next-bar-open execution convention.** Orders generated on bar `t` are filled at the open of bar `t` (same bar) using the bar's open price. This is a simplification relative to a true next-bar model but more realistic than close-to-close execution. Slippage is applied directionally: buys fill above open, sells below.

**Proportional cost model.** Commission and slippage are specified in basis points of gross notional. This is the standard representation used in institutional desks and makes costs easy to calibrate from broker disclosures. One basis point is 0.01%.

**Repository pattern.** Database access is encapsulated in repository classes (`OHLCVRepository`, `BacktestRepository`, `WalkForwardRepository`). These accept a SQLAlchemy `Session` and return domain objects or raise `RepositoryError`. The API routes obtain sessions via a context manager. Business logic never constructs SQL queries.

---

## Mathematical Methodology

### Performance metrics

All metrics are computed in `src/backtesting/metrics.py` on a daily return series.

**Annualised return.** Computed as the compound annualised growth rate:

    CAGR = (∏(1 + r_t))^(252/T) - 1

where `T` is the number of bars and 252 is the assumed trading days per year.

**Sharpe ratio.** Annualised excess return divided by annualised standard deviation:

    Sharpe = (CAGR - r_f) / (σ_daily · √252)

The risk-free rate `r_f` is configurable (default 4%). This is the standard ex-post Sharpe and does not account for autocorrelation of returns.

**Sortino ratio.** Replaces total standard deviation with downside deviation (standard deviation of negative returns only):

    Sortino = (CAGR - r_f) / (σ_downside · √252)

This penalises only negative volatility, making it more appropriate for strategies with non-symmetric return distributions.

**Maximum drawdown.** Defined as the maximum peak-to-trough decline in equity:

    MDD = min_t { (E_t - max_{s≤t} E_s) / max_{s≤t} E_s }

**Historical VaR and CVaR.** Computed empirically from the return distribution. VaR at confidence level α is the (1-α) quantile of the return distribution. CVaR (Expected Shortfall) is the mean of all returns below VaR:

    VaR_α = -F^(-1)(1-α)
    CVaR_α = -E[R | R ≤ -VaR_α]

CVaR is a coherent risk measure (satisfying sub-additivity) and is preferred over VaR for tail risk characterisation.

### Monte Carlo simulation

Two sampling methods are implemented:

**Parametric.** Returns are drawn from N(μ, σ²) where μ and σ are estimated from historical data. This is fast but underestimates tail risk if returns are non-normal.

**Block bootstrap.** Historical returns are resampled in overlapping blocks of fixed length. This preserves the empirical distribution shape, including autocorrelation, skewness, and kurtosis. Block size controls the trade-off between preserving serial dependence and sampling diversity.

For each simulation, cumulative paths are constructed by compounding sampled returns:

    V_t = V_0 · ∏_{k=1}^{t} (1 + r_k)

Terminal VaR and CVaR are computed from the distribution of terminal values across all paths.

### Beta estimation

Beta is estimated via OLS regression of portfolio returns on benchmark returns:

    β = Cov(R_p, R_b) / Var(R_b)

The regression variant is used by default because it provides diagnostics (R², p-value, standard error). The covariance-ratio formulation gives identical results and is available as an alternative.

### Walk-forward validation

The validation methodology follows the rolling window approach:

1. Select the first `train_window_days` bars as the in-sample period.
2. Grid-search strategy hyperparameters on the in-sample period, selecting the configuration with the highest Sharpe ratio.
3. Evaluate the selected parameters on the next `test_window_days` bars (out-of-sample).
4. Step forward by `step_days` and repeat.
5. Concatenate all out-of-sample return series to produce an unbiased performance estimate.

Parameter stability is measured as the coefficient of variation (CV = σ/|μ|) of each numeric hyperparameter across windows. High CV (>0.3 by default) suggests the parameter is not robustly identifiable from the data, which is a sign of overfitting or insufficient signal.

---

## Folder Structure

```
qb-risk-infra/
├── src/
│   ├── core/                  # Shared infrastructure
│   │   ├── config.py          # Pydantic settings model
│   │   ├── logging.py         # JSON / text log formatters
│   │   ├── exceptions.py      # Domain exception hierarchy
│   │   └── types.py           # Shared dataclasses (Order, Fill, etc.)
│   ├── ingestion/
│   │   ├── base.py            # BaseIngestionAdapter abstract class
│   │   ├── csv_adapter.py     # CSV-based OHLCV loading
│   │   ├── api_adapter.py     # Alpha Vantage HTTP adapter
│   │   └── validator.py       # Structural OHLCV validation
│   ├── strategies/
│   │   ├── base.py            # BaseStrategy + Signal
│   │   ├── moving_average_crossover.py
│   │   ├── mean_reversion.py
│   │   ├── momentum.py
│   │   ├── volatility_breakout.py
│   │   └── registry.py        # Strategy lookup and registration
│   ├── backtesting/
│   │   ├── engine.py          # Event-driven backtest runner
│   │   ├── order_router.py    # Slippage + commission simulation
│   │   ├── portfolio.py       # Cash, position, P&L tracking
│   │   └── metrics.py        # Sharpe, Sortino, MDD, VaR, CVaR, etc.
│   ├── walkforward/
│   │   └── validator.py       # Rolling window train/test + grid search
│   ├── montecarlo/
│   │   └── simulator.py       # Parametric + bootstrap MC paths
│   ├── risk/
│   │   └── analytics.py       # Rolling vol, VaR, CVaR, beta, correlation
│   ├── explainability/
│   │   ├── feature_builder.py # Technical feature engineering
│   │   └── shap_analyzer.py   # GBT surrogate model + permutation importance
│   ├── db/
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   ├── repository.py      # Repository pattern implementations
│   │   └── session.py         # Engine + session factory
│   ├── api/
│   │   ├── main.py            # FastAPI app factory + lifespan
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── routes/
│   │       ├── health.py      # /health/ping, /health/
│   │       ├── backtest.py    # /api/v1/backtest/run
│   │       ├── simulation.py  # /api/v1/simulation/walkforward, /montecarlo
│   │       └── analytics.py   # /api/v1/analytics/risk, /runs
│   └── dashboard/
│       └── app.py             # Streamlit analytical dashboard
├── tests/
│   ├── unit/
│   │   ├── test_metrics.py
│   │   ├── test_strategies.py
│   │   ├── test_validator.py
│   │   ├── test_backtest_engine.py
│   │   ├── test_monte_carlo.py
│   │   └── test_risk_analytics.py
│   └── integration/           # Requires running PostgreSQL + Redis
├── config/
│   ├── config.yaml            # Default configuration
│   └── logging.yaml           # Log handler configuration
├── scripts/
│   ├── generate_sample_data.py  # GBM synthetic OHLCV generator
│   └── init_db.sql            # PostgreSQL init script
├── data/raw/                  # CSV files go here (gitignored)
├── logs/                      # Log output (gitignored)
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── Makefile
└── README.md
```

---

## API

The FastAPI application is served at `http://localhost:8000`. Interactive documentation is available at `/docs` (Swagger UI) and `/redoc`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/` | Full health check with DB + Redis connectivity |
| GET | `/health/ping` | Liveness probe |
| POST | `/api/v1/backtest/run` | Execute a strategy backtest |
| GET | `/api/v1/backtest/strategies` | List registered strategies |
| POST | `/api/v1/simulation/walkforward` | Run walk-forward validation |
| POST | `/api/v1/simulation/montecarlo/from-returns` | Run MC simulation from returns |
| POST | `/api/v1/analytics/risk` | Compute risk metrics for a run |
| GET | `/api/v1/analytics/runs` | List recent backtest runs |

All endpoints accept and return JSON. Request and response shapes are defined in `src/api/schemas.py` and automatically documented in the OpenAPI schema.

---

## Deployment

### Local development (no Docker)

```bash
# 1. Clone and set up
git clone https://github.com/wittyswayam/qb-risk-infra
cd qb-risk-infra
make install-dev

# 2. Configure
cp .env.example .env
# Edit .env: set database password, optionally Alpha Vantage key

# 3. Generate sample data
python scripts/generate_sample_data.py --output data/raw

# 4. Start the API server
make api

# 5. Start the dashboard (separate terminal)
make dashboard
```

The API will be at `http://localhost:8000/docs`. The dashboard at `http://localhost:8501`.

### Docker Compose (recommended)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with real values

# Generate sample data on the host
python scripts/generate_sample_data.py --output data/raw

# Start the full stack
make docker-up

# Tail logs
make docker-logs

# Tear down
make docker-down
```

This starts PostgreSQL, Redis, the FastAPI API, and the Streamlit dashboard as separate containers connected on a shared bridge network.

### Environment variables

All configuration can be overridden via environment variables using the `QB__` prefix with double-underscore nesting. See `.env.example` for the full reference. The application reads `.env` automatically at startup via `python-dotenv`.

---

## Testing

```bash
# Full unit test suite with coverage
make test

# Run without integration tests (faster, no DB required)
make test-fast

# Generate HTML coverage report
make test-cov-html
open htmlcov/index.html

# Run a specific test file
python -m pytest tests/unit/test_strategies.py -v

# Run integration tests (requires running PostgreSQL + Redis)
make test-integration
```

Unit tests cover:
- All strategy implementations and the signal/registry system
- The backtesting engine, order router, and portfolio tracker
- All performance metric computations (Sharpe, Sortino, MDD, VaR, CVaR)
- OHLCV data validation
- Monte Carlo simulation (shape, convergence, stress testing)
- Risk analytics (rolling vol, VaR, beta estimation, correlation)

Tests use deterministic random seeds and synthetic data generated inline. No external data files are required to run the unit suite.

---

## Limitations and Assumptions

**Single-symbol backtesting.** The engine processes one symbol at a time. Multi-symbol portfolios require running engines in parallel and combining results at the portfolio level. Cross-asset correlation during execution is not modelled.

**Daily bar resolution.** The primary design target is daily OHLCV data. Intraday data is supported via the CSV adapter and resampling, but the execution model (next-open convention) is designed for daily granularity. Intraday microstructure effects are not modelled.

**Proportional cost model.** The slippage model assumes slippage scales linearly with price, independent of trade size and market liquidity. This is a reasonable approximation for liquid large-cap equities at modest position sizes. Market impact is not modelled.

---

## Extending the System

**Adding a new strategy.** Subclass `BaseStrategy` in `src/strategies/`, implement `on_bar`, `reset`, `required_history`, and `get_params`, then register it with `register_strategy("your_name", YourClass)` in `src/strategies/registry.py`. The backtesting engine, walk-forward validator, and API will all pick it up automatically.

**Adding a new ingestion adapter.** Subclass `BaseIngestionAdapter` in `src/ingestion/`, implement `fetch` and `available_symbols`. Pass the adapter instance to `BacktestEngine.run` or the API route handler.

**Adding a new API route.** Create a router file in `src/api/routes/`, define endpoints, and include the router in `src/api/main.py`. Add corresponding Pydantic schemas to `src/api/schemas.py`.

---

## Dependencies

Core runtime dependencies and the rationale for each:

- **FastAPI + uvicorn**: Async-capable API framework with automatic OpenAPI documentation.
- **pandas + numpy**: Industry-standard dataframe and numerical computing layer for time-series operations.
- **scipy**: Statistical functions (regression, distribution fitting) used in beta estimation and risk metrics.
- **scikit-learn**: GradientBoostingClassifier for the explainability surrogate model; pipeline and permutation importance utilities.
- **SQLAlchemy 2.x**: ORM with repository pattern support. The 2.x API uses `Session.execute` with `select()` statements rather than legacy query methods.
- **pydantic v2**: Data validation for configuration and API schemas. The v2 API uses `model_validator` and `field_validator` decorators.
- **plotly + streamlit**: Interactive visualisation in the analytical dashboard.
- **redis**: Available for caching computed results in high-throughput scenarios (not yet used in core path).

Optional:
- **shap**: SHAP TreeExplainer values for the surrogate model. The system degrades gracefully to permutation importance if not installed.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main` and `develop`:

1. **Quality job**: Checks black formatting and ruff linting.
2. **Test job**: Runs the unit test suite across Python 3.11 and 3.12 with coverage reporting.
3. **Docker build job**: Validates the Dockerfile compiles to a runnable image.
4. **Integration job**: Runs against live PostgreSQL and Redis service containers (main branch pushes only).

All jobs use pip caching to minimise cold-start time. The test matrix ensures compatibility across minor Python versions before they become the production default.

---

## Licence

MIT. See `LICENSE` for details.
