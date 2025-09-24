import os
import shutil
import subprocess
import sys
from datetime import datetime
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import optuna
import pandas as pd
import polars as pl
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import mean_absolute_percentage_error

from vercye_ops.matching_sim_real.utils import load_simulation_data

# PARAMS
BASE_APSIM_PATH = "/home/rohan/nasa-harvest/vercye/vercye_ops/vercye_ops/apsim/template.apsimx"  # This needs to be a prepared APSIM file with the placeholders

matching_file_dir = "/home/rohan/nasa-harvest/vercye/experiments/synthetic_cultivars"  # private script
poltavaFields23_dict = {
    "reported_mean_yield": 6659.597315436242,
    "mean_sowing_date": "26-sep",
    "rs_mean_lai": pd.read_csv("/home/rohan/nasa-harvest/vercye/vercye_ops/vercye_ops/apsim/mean_rs_lai.csv"),
    "mean_met_path": "/home/rohan/nasa-harvest/vercye/vercye_ops/vercye_ops/apsim/mean_metdata_poltava.met",
    "simulation_start": "2022-08-01",
    "simulation_end": "2023-08-01",
}

loss = "yield_and_lai_pareto"
use_matching = True
experiment_dict = poltavaFields23_dict

study_name = "cultivar_search"
journal_file_path = "optuna.journal"
out_dir = None
ntrials = 10000
cores = 1

# Typically dont change from here
mean_sowing_date = experiment_dict["mean_sowing_date"]
reported_mean_yield = experiment_dict["reported_mean_yield"]
rs_mean_lai = experiment_dict["rs_mean_lai"]
mean_met_path = experiment_dict["mean_met_path"]
sim_start = experiment_dict["simulation_start"]
sim_end = experiment_dict["simulation_end"]

sys.path.append(matching_file_dir)
from matching import match_simulations  # noqa: E402

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


# Experiment 1: Take all simulation results, build mean LAI TS & mean max yield from all simulations and compute loss
# Experiment 2: Take only matched simulations, build mean LAI TS & mean max yield from those and compute loss.
# Experiment 3: Compare pareto solution vs combined solution with both weighted at 0.5
# Experiment 4: Use yield only for loss, since matching already was responsible for identifying close lai series

# Experiments setup A
# Select year
# Compute mean_rs_lai, mean_reported_yield, mean_sowing_date, mean_met for that year
# Run optimization
# Validate improvement on other years, by comparing improvement in mape/R2/RRMSE


# Experiments setup B
# Select multiple years (leave one out) with k-folds
# Compute each years mean_rs_lai, mean_reported_yield, mean_sowing_date, mean_met
# Run for each year an optimization and then take mean value of all found genotypical params
# Validate improvement to generic config on leaft out year


def optimization_objective(
    trial,
    base_apsim_file_path,
    reported_yield,
    rs_lai_ts,
    mean_sowing_date,
    mean_met_path,
    output_dir,
    use_matching,
    loss_fn,
    sim_start,
    sim_end,
):
    # controls yield potential. Higher values mean more grains per unit stem biomass.
    grain_per_gram_of_stem = trial.suggest_int("grain_per_gram_of_stem", 0, 100, step=10)

    # controls LAI peak timing. Smaller values mean leaves appear faster, canopy closes earlier.
    base_phyllocron = trial.suggest_int("base_phyllocron", 1, 100, step=9)

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

    os.makedirs(
        os.path.join(output_dir, f"{grain_per_gram_of_stem}_{base_phyllocron}_{vegetative_phase}_{ppln}"), exist_ok=True
    )
    new_mean_met_path = os.path.join(
        output_dir, f"{grain_per_gram_of_stem}_{base_phyllocron}_{vegetative_phase}_{ppln}", "mean_metdata.met"
    )
    shutil.copy(mean_met_path, new_mean_met_path)
    updated_apsim_file_path = os.path.join(
        output_dir,
        f"{grain_per_gram_of_stem}_{base_phyllocron}_{vegetative_phase}_{ppln}",
        f"{grain_per_gram_of_stem}_{base_phyllocron}_{vegetative_phase}_{ppln}.apsimx",
    )
    update_apsim_file(
        base_apsim_file_path,
        param_dictionary,
        mean_sowing_date,
        new_mean_met_path,
        updated_apsim_file_path,
        sim_start,
        sim_end,
    )
    run_simulation(updated_apsim_file_path)

    apsim_report_db_path = updated_apsim_file_path.replace(".apsimx", ".db")
    apsim_report_db = load_apsim_db(apsim_report_db_path)

    if use_matching:
        apsim_report_db = filter_matched_simulations(rs_lai_ts, apsim_report_db)

    predicted_mean_yield = get_predicted_mean_yield(apsim_report_db)
    predicted_mean_lai_ts = get_predicted_mean_lai_ts(apsim_report_db)

    # Align dates of predicted and true to keep only range from RS
    rs_lai_ts["Date"] = pd.to_datetime(rs_lai_ts["Date"])
    predicted_mean_lai_ts["Date"] = pd.to_datetime(predicted_mean_lai_ts["Date"])
    aligned = rs_lai_ts.merge(predicted_mean_lai_ts, on="Date", how="inner").dropna()
    rs_lai_vals = aligned["rs_mean_lai"].to_numpy()
    pred_lai_vals = aligned["pred_mean_lai"].to_numpy()

    return loss_fn(reported_yield, predicted_mean_yield, rs_lai_vals, pred_lai_vals)


def filter_matched_simulations(rs_lai_ts: pd.DataFrame, report_db: pd.DataFrame):
    good_sim_ids = match_simulations(rs_lai_ts, report_db, "Wheat", 0.2, 0.8, 0.2, 0.9, 4, True).index
    return report_db[report_db["SimulationID"].isin(good_sim_ids)]


def compute_yield_loss(reported_yield, predicted_yield):
    return mean_absolute_percentage_error([reported_yield], [predicted_yield])


def compute_lai_ts_loss(rs_lai_ts, predicted_lai_ts):
    return mean_absolute_percentage_error(rs_lai_ts, predicted_lai_ts)


def estimate_cultivars(
    study_name,
    journal_file_path,
    n_trials,
    reported_yield,
    rs_lai_ts,
    mean_sowing_date,
    mean_met_path,
    output_dir,
    use_matching,
    loss_fn,
    metric_names,
    loss_direction,
    sim_start,
    sim_end,
):
    # Initiate study with shared backend throughout all parallel processes
    # for sharing information on which parameters to explore
    study = optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(file_path=journal_file_path)),
        load_if_exists=True,
        directions=loss_direction,
    )

    study.set_metric_names(metric_names)

    study.optimize(
        partial(
            optimization_objective,
            base_apsim_file_path=BASE_APSIM_PATH,
            reported_yield=reported_yield,
            rs_lai_ts=rs_lai_ts,
            mean_sowing_date=mean_sowing_date,
            mean_met_path=mean_met_path,
            output_dir=output_dir,
            use_matching=use_matching,
            loss_fn=loss_fn,
            sim_start=sim_start,
            sim_end=sim_end,
        ),
        n_trials=n_trials,
    )
    return study


def update_apsim_file(
    base_apsim_file_path: str,
    param_dictionary: Dict[str, Any],
    mean_sowing_date: str,
    mean_met_path: str,
    output_apsim_file_path: str,
    sim_start: str,
    sim_end: str,
):
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

    # find all occurences of SOWINGDATEPLACEHOLDER and replace with mean_sowing_date
    content = content.replace("SOWINGDATEPLACEHOLDER", mean_sowing_date)
    content = content.replace("METPLACEHOLDER", os.path.abspath(mean_met_path))
    content = content.replace("SIMSTARTPLACEHOLDER", sim_start)
    content = content.replace("SIMENDPLACEHOLDER", sim_end)

    with open(output_apsim_file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_apsim_file_path


def run_simulation(apsim_file_path: str):
    """Run the APSIM simulator with a provided configfile."""
    # subprocess.run(
    #     ["/gpfs/data1/cmongp2/wronk/Builds/ApsimX/bin/Release/net6.0/Models", apsim_file_path, "--cpu-count", "8"],
    #     cwd=Path(apsim_file_path).parent
    # )

    run_dir = Path(apsim_file_path).parent
    run_dir = os.path.abspath(run_dir)
    print(
        " ".join(
            ["docker", "run", "-v", f"{run_dir}:{run_dir}", "apsiminitiative/apsimng", os.path.abspath(apsim_file_path)]
        )
    )

    subprocess.run(
        ["docker", "run", "-v", f"{run_dir}:{run_dir}", "apsiminitiative/apsimng", os.path.abspath(apsim_file_path)],
        cwd=run_dir,
        check=True,
        start_new_session=True,
    )


def get_predicted_mean_lai_ts(report_data_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a mean predicted LAI timeseries.

    Built from taking the mean LAI value per date over all simulations.
    Assumes a 'Date' column exists.
    """
    mean_lai_ts = (
        report_data_df.groupby("Date")["Wheat.Leaf.LAI"].mean().reset_index()  # ensure "Date" is a column, not an index
    )
    mean_lai_ts["Date"] = pd.to_datetime(mean_lai_ts["Date"])
    mean_lai_ts.rename(columns={"Wheat.Leaf.LAI": "pred_mean_lai"}, inplace=True)
    return mean_lai_ts


def get_predicted_mean_yield(report_data_df: pd.DataFrame) -> float:
    """Computes the mean predicted yield, by taking the maximum predicted yield per simulation."""
    max_yields = report_data_df.groupby("SimulationID")["Yield"].max()
    return max_yields.mean()


def load_rs_lai_timeseries(rs_lai_path: str) -> pd.DataFrame:
    df = pd.read_csv(rs_lai_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Median LAI Adjusted"]].rename(columns={"Median LAI Adjusted": "rs_mean_lai"})


def get_mean_rs_lai_timeseries(rs_lai_file_paths: list[str]) -> pl.DataFrame:
    dfs = [load_rs_lai_timeseries(path) for path in rs_lai_file_paths]

    # Rename "Median LAI Adjusted" in each df to avoid collisions
    renamed = [df.rename({"Median LAI Adjusted": f"lai_{i}"}) for i, df in enumerate(dfs)]

    combined = renamed[0]
    for df in renamed[1:]:
        combined = combined.join(df, on="Date", how="outer")

    combined = combined.with_columns(
        pl.mean_horizontal([col for col in combined.columns if col.startswith("lai_")]).alias("rs_mean_lai")
    )
    return combined.select(["Date", "rs_mean_lai"])


def get_mean_reported_yield(yield_file: str) -> float:
    """Compute the mean yield in kg/ha from all regions in the file."""
    df = pd.read_csv(yield_file)
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


def load_met_file(met_file_path: str):
    header_lines = (8,)
    column_line = (6,)
    with open(met_file_path, "r") as f:
        lines = f.readlines()

    # Extract metadata
    metadata = {
        "latitude": float(lines[1].split("=")[1].strip().split()[0]),
        "longitude": float(lines[2].split("=")[1].strip().split()[0]),
        "tav": float(lines[3].split("=")[1].strip().split()[0]),
        "amp": float(lines[4].split("=")[1].strip().split()[0]),
    }

    # Extract column names
    columns = lines[column_line - 1].strip().split()

    # Extract units
    units_line = lines[column_line].strip().split()
    units = dict(zip(columns, units_line))
    metadata["units"] = units

    # Read the data
    data = [line.strip().split() for line in lines[header_lines:]]

    # Create DataFrame, convert types
    df = pd.DataFrame(data, columns=columns)
    MET_FILE_DTYPES = {
        "year": int,
        "day": int,
        "radn": float,
        "maxt": float,
        "mint": float,
        "meant": float,
        "rain": float,
        "wind": float,
        "data_type": str,
    }
    df = df.astype(MET_FILE_DTYPES)
    df["date"] = pd.to_datetime(df["year"].astype(str) + df["day"].astype(str), format="%Y%j")

    return df


def build_mean_metdata_file(metdata_files: List[str], output_path: str):
    all_dfs = [load_met_file(met_file_path) for met_file_path in metdata_files]

    df_all = pd.concat(all_dfs, ignore_index=True)

    df_mean = df_all.groupby("date", as_index=False).mean(numeric_only=True)

    df_mean["year"] = df_mean["date"].dt.year
    df_mean["day"] = df_mean["date"].dt.dayofyear.astype(int)

    # Reorder columns in original order
    cols_order = ["year", "day", "radn", "maxt", "mint", "meant", "rain", "wind"]
    df_out = df_mean[cols_order]

    metadata = {
        "latitude": all_dfs[0]["latitude"] if "latitude" in all_dfs[0] else None,
        "longitude": all_dfs[0]["longitude"] if "longitude" in all_dfs[0] else None,
        "tav": all_dfs[0]["tav"] if "tav" in all_dfs[0] else None,
        "amp": all_dfs[0]["amp"] if "amp" in all_dfs[0] else None,
        "units": all_dfs[0]["units"] if "units" in all_dfs[0] else None,
    }

    with open(output_path, "w") as f:
        f.write("! Mean MET data file\n")
        f.write(f"latitude = {metadata['latitude']}\n")
        f.write(f"longitude = {metadata['longitude']}\n")
        f.write(f"tav = {metadata['tav']}\n")
        f.write(f"amp = {metadata['amp']}\n")
        f.write("!\n")

        # Write header
        f.write(" ".join(cols_order) + "\n")

        # Write units line
        units = metadata["units"]
        unit_line = " ".join(units.get(c, "-") for c in cols_order)
        f.write(unit_line + "\n")

        # Write data rows
        for _, row in df_out.iterrows():
            f.write(
                f"{int(row['year']):4d} "
                f"{int(row['day']):3d} "
                f"{row['radn']:6.1f} "
                f"{row['maxt']:5.1f} "
                f"{row['mint']:5.1f} "
                f"{row['meant']:6.1f} "
                f"{row['rain']:6.1f} "
                f"{row['wind']:6.1f}\n"
            )

    return output_path


def loss_fn_yield_lai_pareto(reported_yield, predicted_mean_yield, rs_lai_vals, pred_lai_vals):
    yield_loss = compute_yield_loss(reported_yield, predicted_mean_yield)
    lai_loss = compute_lai_ts_loss(rs_lai_vals, pred_lai_vals)
    return yield_loss, lai_loss


def loss_fn_yield(reported_yield, predicted_mean_yield, rs_lai_vals, pred_lai_vals):
    yield_loss = compute_yield_loss(reported_yield, predicted_mean_yield)
    return yield_loss


def loss_fn_yield_lai_weighted(reported_yield, predicted_mean_yield, rs_lai_vals, pred_lai_vals):
    yield_loss = compute_yield_loss(reported_yield, predicted_mean_yield)
    lai_loss = compute_lai_ts_loss(rs_lai_vals, pred_lai_vals)
    return yield_loss, lai_loss


# @click.command()
# @click.option("--cores", type=int, default=1, help="Number of cores to use")
# @click.option("--ntrials", type=int, default=2000, help="Number of trials in parameter search.")
# @click.option("--study-name", default="cultivar_search", show_default=True, help="Parameter search study name.")
# @click.option("--output-dir", type=str,  help="Output folder")
# @click.option("--use-matching", is_flag=True, help="Whether to only consider simulations that would be matched by vercyes matching algorithm for computing the losses.")
# def main(cores: int, ntrials: int, study_name: str, output_dir: str, use_matching: str):


def main():
    base = ntrials // max(1, cores)
    remainder = ntrials % max(1, cores)
    trials_for_worker = [base + (1 if i < remainder else 0) for i in range(cores)]

    if not out_dir:
        output_dir = str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
        os.makedirs(output_dir)
    else:
        output_dir = out_dir

    # reported_mean_yield = get_mean_reported_yield("/gpfs/data1/cmongp2/vercye/data/yieldstudies/PoltavaFields2023_08092025/PoltavaFields2023_08092025/2023/referencedata_primary-2023.csv")
    # rs_lai_paths = glob("/gpfs/data1/cmongp2/vercye/data/yieldstudies/MykolayivFields24_20250905/MykolayivFields24_20250905/2024/T-0/**/*_LAI_STATS.csv", recursive=True)
    # rs_mean_lai = get_mean_rs_lai_timeseries(rs_lai_paths)
    # geojsons = glob("/gpfs/data1/cmongp2/vercye/data/yieldstudies/MykolayivFields24_20250905/MykolayivFields24_20250905/2024/T-0/**/*.geojson", recursive=True)
    # geojsons = [f for f in geojsons if 'agg' not in f]
    # mean_sowing_date = get_mean_sowing_date_apsimformat(geojsons, 'Sow_Date')
    # metdata_files = glob("/gpfs/data1/cmongp2/vercye/data/yieldstudies/PoltavaFields2023_08092025_SowDate_Fixed/PoltavaFields2023_08092025_SowDate_Fixed/2023/T-0/**/*.met", recursive=True")
    # mean_met_path = build_mean_metdata_file(metdata_files)

    losses_dict = {
        "yield": (loss_fn_yield, ["yield_mape"], ["minimize"]),
        "yield_and_lai_pareto": (loss_fn_yield_lai_pareto, ["yield_mape", "lai_mape"], ["minimize", "minimize"]),
        "yield_and_lai_weighted05": (loss_fn_yield_lai_weighted, ["05yieldmape+05laimape"], ["minimize"]),
    }

    loss_fn, metric_names, loss_direction = losses_dict[loss]

    with get_context("spawn").Pool(processes=cores, maxtasksperchild=1) as pool:
        try:
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
                        mean_met_path,
                        output_dir,
                        use_matching,
                        loss_fn,
                        metric_names,
                        loss_direction,
                        sim_start,
                        sim_end,
                    )
                    for i in range(cores)
                ],
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

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
