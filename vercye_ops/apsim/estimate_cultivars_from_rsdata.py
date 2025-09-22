import subprocess
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict

import click
import geopandas as gpd
import optuna
import polars as pl
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import mean_absolute_percentage_error

from vercye_ops.matching_sim_real.utils import load_simulation_data

# Not included in dependencies currently
# pip install optuna optuna-dashboard polars

# This experiment can only be run after a normal vercye run and collects all files it needs from this run.
# The new APSIM config can then be used for a new run after.

# Some thoughts
# Would also be an interesting experiement instead of estimating the genotypical parameters to
# use all exsting cultivars from APSIM and identify those that are closest to our objective

# Alternative option for matching loss
# Use true matching algorithm to filter out simulations and take mean of remainings RS & LAI curve

# We would likely also want to explore single objective that combined both and multi objective both
# instead of going for the pareto optimal solution


# Experiment 1: Take all simultion resoluts, build mean LAI TS & mean max yield from all simulations and compute loss
# Experiment 2: Take only matched simulations, build mean LAI TS & mean max yield from those and compute loss.


def optimization_objective(trial, base_apsim_file_path, reported_yield, rs_lai_ts, mean_sowing_date):
    # controls yield potential. Higher values mean more grains per unit stem biomass.
    grain_per_gram_of_stem = trial.suggest_int("grain_per_gram_of_stem", 0, 100, step=10)

    # controls LAI peak timing. Smaller values mean leaves appear faster, canopy closes earlier.
    base_phyllocron = trial.suggest_int("base_phyllocron", 0, 100, step=10)

    # influences canopy light interception, hence the magnitude of LAI.
    vegetative_phase = trial.suggest_int("vegetative_phase", 0, 1)

    # affects final leaf number and thereby both LAI peak value and timing.
    ppln = trial.suggest_int("ppln", 0, 1)

    param_dictionary = {
        "[Grain].NumberFunction.GrainNumber.GrainsPerGramOfStem.FixedValue": grain_per_gram_of_stem,
        "[Phenology].Phyllochron.BasePhyllochron.FixedValue": base_phyllocron,
        "[Leaf].ExtinctionCoeff.VegetativePhase.FixedValue": vegetative_phase,
        "[Phenology].CAMP.FLNparams.PpLN": ppln,
    }

    updated_apsim_file_path = None
    update_apsim_file(base_apsim_file_path, param_dictionary, updated_apsim_file_path)
    run_simulation(updated_apsim_file_path)

    apsim_report_db = load_apsim_db()
    predicted_mean_lai_ts = get_predicted_mean_lai_ts(apsim_report_db)
    predicted_mean_yield = get_predicted_mean_yield(apsim_report_db)

    # Align dates of predicted and true to keep only range from RS
    aligned = rs_lai_ts.join(predicted_mean_lai_ts, on="Date", how="inner").drop_nulls()
    rs_lai_vals = aligned["mean_lai"].to_numpy()
    pred_lai_vals = aligned["Mean_LAI"].to_numpy()

    lai_ts_loss = compute_lai_ts_loss(rs_lai_vals, pred_lai_vals)
    yield_loss = compute_yield_loss(reported_yield, predicted_mean_yield)
    return lai_ts_loss, yield_loss


def compute_yield_loss(reported_yield, predicted_yield):
    return mean_absolute_percentage_error(reported_yield, predicted_yield)


def compute_lai_ts_loss(rs_lai_ts, predicted_lai_ts):
    return mean_absolute_percentage_error(rs_lai_ts, predicted_lai_ts)


def estimate_cultivars(study_name, journal_file_path, n_trials, reported_yield, rs_lai_ts, mean_sowing_date):
    # Initiate study with shared backend throughout all parallel processes
    # for sharing information on which parameters to explore
    study = optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(file_path=journal_file_path)),
        load_if_exists=True,
        directions=["minimize", "minimize"],
    )
    study.optimize(
        partial(
            optimization_objective,
            base_apsim_file_path="path/to/base.apsim",
            reported_yield=reported_yield,
            rs_lai_ts=rs_lai_ts,
            mean_sowing_date=mean_sowing_date,
        ),
        n_trials=n_trials,
    )
    return study


def update_apsim_file(base_apsim_file_path: str, param_dictionary: Dict[str, Any], output_apsim_file_path: str):
    """Inject geonotypical parameters into APSIM file

    Params:
        base_apsim_file_path: Path to an APSIM 'template' file that contains placeholder values set to -1 for each entry in param_dictionary.
        param_dictionary: Dictionary containing full names of parameters in APSIM file, and the new value.
        output_apsim_file_path: Path to the new updated APSIM file.

    Returns:
        Path to the new APSIM file with updated parameters
    """
    with open(base_apsim_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    for param_name, param_value in param_dictionary.items():
        content = content.replace(f"{param_name} = -1", f"{param_name} = {str(param_value)}")

    with open(output_apsim_file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_apsim_file_path


def run_simulation(apsim_file_path: str):
    """Run the APSIM simulator with a provided configfile."""
    subprocess.run(
        ["/gpfs/data1/cmongp2/wronk/Builds/ApsimX/bin/Release/net6.0/Models", apsim_file_path, "--cpu-count", "8"]
    )


def get_predicted_mean_lai_ts(report_data_df: pl.DataFrame) -> pl.DataFrame:
    """Compute a mean predicted LAI timeseries.

    Built from taking the mean LAI value per date over all simulations.
    Assumes a 'Date' column exists.
    """
    mean_lai_ts = report_data_df.groupby("Date").agg(pl.col("Wheat.Leaf.LAI").mean().alias("Mean_LAI")).sort("Date")
    return mean_lai_ts


def get_predicted_mean_yield(report_data_df: pl.DataFrame) -> float:
    """Computes the mean predicted yield, by taking the maximum predicted yield per simulation."""
    max_yields = report_data_df.groupby("SimulationID").agg(pl.col("Yield").max().alias("Max_Yield"))
    return max_yields["Max_Yield"].mean()


def load_rs_lai_timeseries(rs_lai_path: str) -> pl.DataFrame:
    df = pl.read_csv(rs_lai_path, try_parse_dates=True)
    return df.select(["Date", "Median LAI Adjusted"])


def get_mean_rs_lai_timeseries(rs_lai_file_paths: list[str]) -> pl.DataFrame:
    dfs = [load_rs_lai_timeseries(path) for path in rs_lai_file_paths]

    # Rename "Median LAI Adjusted" in each df to avoid collisions
    renamed = [df.rename({"Median LAI Adjusted": f"lai_{i}"}) for i, df in enumerate(dfs)]

    combined = renamed[0]
    for df in renamed[1:]:
        combined = combined.join(df, on="Date", how="outer")

    combined = combined.with_columns(
        pl.mean_horizontal([col for col in combined.columns if col.startswith("lai_")]).alias("mean_lai")
    )
    return combined.select(["Date", "mean_lai"])


def get_mean_reported_yield(yield_file: str) -> float:
    """Compute the mean yield in kg/ha from all regions in the file."""
    df = pl.read_csv(yield_file)
    return df["reported_mean_yield_kg_ha"].mean()


def get_mean_sowing_date_apsimformat(geojson_paths: list[str], sowing_date_col: str) -> str:
    """Compute the mean of all sowing dates and return it as a string formatted for APSIM (e.g. 15-sep)."""
    all_sowing_dates = []
    for geojson_path in geojson_paths:
        gdf = gpd.read_file(geojson_path)
        sowing_date = datetime.strptime(gdf[sowing_date_col].values[0], "%Y-%m-%d")
        all_sowing_dates.append(sowing_date)

    mean_timestamp = sum(d.timestamp() for d in all_sowing_dates) / len(all_sowing_dates)
    mean_date = datetime.fromtimestamp(mean_timestamp)
    return mean_date.strftime("%-d-%b").lower()


def generate_optimized_apsim_file(best_params):
    pass


def load_apsim_db(db_path: str):
    return load_simulation_data(db_path, "wheat")


@click.command()
@click.option("--cores", type=int, default=4, help="Number of cores to use")
@click.option("--ntrials", type=int, default=100, help="Number of trials in parameter search.")
@click.option("--study-name", default="cultivar_search", show_default=True, help="Parameter search study name.")
def main(cores: int, ntrials: int, study_name: str):
    base = ntrials // max(1, cores)
    remainder = ntrials % max(1, cores)
    trials_for_worker = [base + (1 if i < remainder else 0) for i in range(cores)]

    # reported_mean_yield = get_mean_reported_yield("/gpfs/data1/cmongp2/vercye/data/yieldstudies/PoltavaFields2023_08092025/PoltavaFields2023_08092025/2023/referencedata_primary-2023.csv")
    # rs_lai_paths = glob("/gpfs/data1/cmongp2/vercye/data/yieldstudies/MykolayivFields24_20250905/MykolayivFields24_20250905/2024/T-0/**/*_LAI_STATS.csv", recursive=True)
    # rs_mean_lai = get_mean_rs_lai_timeseries(rs_lai_paths)
    # geojsons = glob("/gpfs/data1/cmongp2/vercye/data/yieldstudies/MykolayivFields24_20250905/MykolayivFields24_20250905/2024/T-0/**/*.geojson", recursive=True)
    # geojsons = [f for f in geojsons if 'agg' not in f]
    # mean_sowing_date = get_mean_sowing_date_apsimformat(geojsons, 'Sow_Date')

    poltavaFields23_dict = {
        "reported_mean_yield": 6659.597315436242,
        "mean_sowing_date": "26-sep",
        "rs_mean_lai": pl.read_csv("/home/rohan/nasa-harvest/vercye/vercye_ops/vercye_ops/apsim/mean_rs_lai.csv"),
    }

    mean_sowing_date = poltavaFields23_dict["mean_sowing_date"]
    reported_mean_yield = poltavaFields23_dict["reported_mean_yield"]
    rs_mean_lai = poltavaFields23_dict["rs_mean_lai"]

    journal_file_path = "optuna.journal"

    with Pool(processes=cores) as pool:
        pool.starmap(
            estimate_cultivars,
            [
                (
                    study_name,
                    journal_file_path,
                    trials_for_worker[i],
                    reported_mean_yield,
                    rs_mean_lai,
                    mean_sowing_date,
                )
                for i in range(cores)
            ],
        )

    storage = JournalStorage(JournalFileBackend(file_path=journal_file_path))
    study = optuna.load_study(study_name=study_name, storage=storage)

    print("Number of Pareto optimal solutions:", len(study.best_trials))
    best_params = []
    for trial in study.best_trials:
        print("Values:", trial.values)
        print("Params:", trial.params)
        best_params.append(trial.params)

    generate_optimized_apsim_file(best_params)


if __name__ == "__main__":
    main()
