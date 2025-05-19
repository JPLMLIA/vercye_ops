## Snakemake Niceties
To visualize DAGs of the pipeline or generate a report, use the following commands:

```
# Visualizing the processing pipeline for a single execution
snakemake --configfile <your_config.yaml> --rulegraph | dot -Tpdf > dag_rulegraph_template.pdf

# Visualizing the graph for all pipeline executions
snakemake --configfile <your_config.yaml> --dag > dag.dot
dot -Tpdf dag.dot > dag_custom.pdf

# Generate a report for a finished pipeline execution (to set graph and runtimes)
snakemake --configfile <your_config.yaml> --report report.html
```

Other helpful command line args are documented on the [snakemake CLI page](https://snakemake.readthedocs.io/en/stable/executing/cli.html).


# Tests
Tests are handled by pytest. Navigate to the root directory and execute `pytest` to run them.