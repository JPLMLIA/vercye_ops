import click
import pandas as pd

from typing import List

from vercye.matching_sim_real.utils import load_simulation_data


def get_best_matches(matches_path: str, num_sims: int):
    sim_matches_data = pd.read_csv(matches_path)

    matched_sims = sim_matches_data[sim_matches_data['StepFilteredOut'].isna()]

    if num_sims:
        matched_sims = matched_sims[:num_sims]

    return matched_sims['SimulationID'].tolist()


def get_sims(db_path: str, ids_to_keep: List[int]):
    query = "SELECT * FROM Report"
    simulatiosn_data = load_simulation_data(db_path, query=query)
    matched_simulations = simulations_data[simulations_data['SimulationID'].isin(ids_to_keep)]
    return matched_simulations


def save_sims(sims: pd.DataFrame, out_path: str):
    sims.to_csv(out_path, index=False)


def extract_and_save_sims(db_path: str, matches_path: str, num_sims: int, out_path: str):
    ids_to_keep = get_best_matches(matches_path, num_sims)

    sim_data_to_keep = get_sims(db_path, ids_to_keep)

    save_sims(sim_data_to_keep, out_path)


@click.command()
@click.option('--db-path', type=click.Path(exists=True), required=True, help='Path to the APSIM DB file')
@click.option('--matches-path', type=click.Path(exists=True), required=True, help='Path to the csv with all matched simulations.', default=None)
@click.option('--num-sims', help='Number of best matched simulation to save. If not specified, will save all.', default=None, type=int)
@click.option('--out_path', type=click.Path(exists=False), required=True, help='Path where to save the simulations (.CSV).')
def main(db_path, matches_path, num_sims, out_path):
    """
    Utility to extract the best n simulations from an APSIM DB file and save those as CSV
    for easier analysis
    """

    extract_and_save_sims(db_path, matches_path, num_sims, out_path)

    print(f'Successfully extracted best simulations to {out_path}.')


if __name__ == '__main__':
    main()