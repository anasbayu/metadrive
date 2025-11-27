"""
Optuna Visualization Script
============================
Script untuk memvisualisasikan hasil optimasi hyperparameter Optuna
Includes: parallel coordinate, optimization history, parameter importance, etc.

Author: Anas Bayu Kusuma
Date: 2025
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_contour,
    plot_slice,
    plot_edf,
)
import plotly.graph_objects as go
import json
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your Optuna study database
STUDY_NAME = "leaky_ppo_metadrive_optuna_1.5M"  # Sesuaikan dengan nama study Anda
STORAGE_PATH = "sqlite:///optuna_leaky_ppo_1.5M.db"  # Path ke database Optuna

# Output directory for plots
OUTPUT_DIR = Path("optuna_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot configuration
PLOT_CONFIG = {
    "width": 1200,
    "height": 600,
    "template": "plotly_white",  # Options: plotly, plotly_white, plotly_dark
}

# ============================================================================
# LOAD OPTUNA STUDY
# ============================================================================

def load_study(study_name: str, storage_path: str) -> optuna.Study:
    """Load existing Optuna study from database"""
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )
        print(f"✅ Study loaded: {study_name}")
        print(f"   Number of trials: {len(study.trials)}")
        print(f"   Best trial: {study.best_trial.number}")
        print(f"   Best value: {study.best_value:.4f}")
        return study
    except KeyError:
        print(f"❌ Study '{study_name}' not found in {storage_path}")
        print("Available studies:")
        studies = optuna.study.get_all_study_summaries(storage=storage_path)
        for s in studies:
            print(f"   - {s.study_name}")
        raise

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_1_optimization_history(study: optuna.Study, output_dir: Path):
    """
    Plot 1: Optimization History
    Shows how the objective value evolves across trials
    """
    print("\n📊 Generating Plot 1: Optimization History...")
    
    fig = plot_optimization_history(study)
    fig.update_layout(
        title="Optimization History (Cumulative Reward vs Trial Number)",
        xaxis_title="Trial Number",
        yaxis_title="Cumulative Reward",
        width=PLOT_CONFIG["width"],
        height=PLOT_CONFIG["height"],
        template=PLOT_CONFIG["template"],
    )
    
    # Add horizontal line for best value
    fig.add_hline(
        y=study.best_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Best: {study.best_value:.2f}",
        annotation_position="right"
    )
    
    # Save
    output_path = output_dir / "01_optimization_history.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved to: {output_path}")
    
    return fig


def plot_2_parallel_coordinate(study: optuna.Study, output_dir: Path):
    """
    Plot 2: Parallel Coordinate Plot
    Shows relationship between hyperparameters and objective value
    Each line represents one trial
    """
    print("\n📊 Generating Plot 2: Parallel Coordinate...")
    
    # Select parameters to display (adjust based on your search space)
    params = [
        "learning_rate",
        "ent_coef",
        "batch_size",
        "n_steps",
        "gamma",
        "gae_lambda",
        "clip_range",
        "n_epochs",
    ]
    
    fig = plot_parallel_coordinate(
        study,
        params=params,
    )
    
    fig.update_layout(
        title="Parallel Coordinate Plot (Hyperparameter Relationships)",
        width=PLOT_CONFIG["width"],
        height=PLOT_CONFIG["height"],
        template=PLOT_CONFIG["template"],
    )
    
    # Save
    output_path = output_dir / "02_parallel_coordinate.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved to: {output_path}")
    
    return fig


def plot_3_param_importances(study: optuna.Study, output_dir: Path):
    """
    Plot 3: Parameter Importance
    Shows which hyperparameters have the most impact on objective value
    Uses fANOVA (functional ANOVA) for importance calculation
    """
    print("\n📊 Generating Plot 3: Parameter Importance...")
    
    try:
        fig = plot_param_importances(study)
        fig.update_layout(
            title="Hyperparameter Importance (fANOVA)",
            xaxis_title="Importance",
            yaxis_title="Hyperparameter",
            width=PLOT_CONFIG["width"],
            height=PLOT_CONFIG["height"],
            template=PLOT_CONFIG["template"],
        )
        
        # Save
        output_path = output_dir / "03_param_importances.html"
        fig.write_html(str(output_path))
        print(f"✅ Saved to: {output_path}")
        
        return fig
    
    except RuntimeError as e:
        print(f"⚠️  Warning: Could not compute parameter importance")
        print(f"   Error: {e}")
        print(f"   This requires ≥3 completed trials")
        return None


def plot_4_contour(study: optuna.Study, output_dir: Path):
    """
    Plot 4: Contour Plot
    Shows 2D relationship between pairs of hyperparameters
    """
    print("\n📊 Generating Plot 4: Contour Plot...")
    
    # Select two most important parameters
    param_pairs = [
        ["learning_rate", "ent_coef"],
        ["gamma", "gae_lambda"],
        ["batch_size", "n_steps"],
    ]
    
    for i, params in enumerate(param_pairs):
        try:
            fig = plot_contour(study, params=params)
            fig.update_layout(
                title=f"Contour Plot: {params[0]} vs {params[1]}",
                width=PLOT_CONFIG["width"],
                height=PLOT_CONFIG["height"],
                template=PLOT_CONFIG["template"],
            )
            
            # Save
            output_path = output_dir / f"04_contour_{i+1}_{params[0]}_vs_{params[1]}.html"
            fig.write_html(str(output_path))
            print(f"✅ Saved to: {output_path}")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not generate contour for {params}")
            print(f"   Error: {e}")


def plot_5_slice(study: optuna.Study, output_dir: Path):
    """
    Plot 5: Slice Plot
    Shows how objective value changes with each hyperparameter independently
    """
    print("\n📊 Generating Plot 5: Slice Plot...")
    
    try:
        fig = plot_slice(study)
        fig.update_layout(
            title="Slice Plot (Individual Parameter Effects)",
            width=PLOT_CONFIG["width"] * 1.5,  # Wider for multiple subplots
            height=PLOT_CONFIG["height"] * 1.2,
            template=PLOT_CONFIG["template"],
        )
        
        # Save
        output_path = output_dir / "05_slice_plot.html"
        fig.write_html(str(output_path))
        print(f"✅ Saved to: {output_path}")
        
        return fig
    
    except Exception as e:
        print(f"⚠️  Warning: Could not generate slice plot")
        print(f"   Error: {e}")
        return None


def plot_6_edf(study: optuna.Study, output_dir: Path):
    """
    Plot 6: Empirical Distribution Function (EDF)
    Shows cumulative distribution of objective values
    """
    print("\n📊 Generating Plot 6: EDF...")
    
    try:
        fig = plot_edf(study)
        fig.update_layout(
            title="Empirical Distribution Function (Objective Value Distribution)",
            xaxis_title="Cumulative Reward",
            yaxis_title="Cumulative Probability",
            width=PLOT_CONFIG["width"],
            height=PLOT_CONFIG["height"],
            template=PLOT_CONFIG["template"],
        )
        
        # Save
        output_path = output_dir / "06_edf.html"
        fig.write_html(str(output_path))
        print(f"✅ Saved to: {output_path}")
        
        return fig
    
    except Exception as e:
        print(f"⚠️  Warning: Could not generate EDF plot")
        print(f"   Error: {e}")
        return None


def plot_7_custom_summary(study: optuna.Study, output_dir: Path):
    """
    Plot 7: Custom Summary Statistics
    Bar plot showing key statistics: completed, pruned, failed trials
    """
    print("\n📊 Generating Plot 7: Custom Summary...")
    
    # Count trial states
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    fig = go.Figure(data=[
        go.Bar(
            x=["Completed", "Pruned", "Failed"],
            y=[completed, pruned, failed],
            text=[completed, pruned, failed],
            textposition="auto",
            marker_color=["green", "orange", "red"]
        )
    ])
    
    fig.update_layout(
        title="Trial Summary Statistics",
        xaxis_title="Trial State",
        yaxis_title="Number of Trials",
        width=PLOT_CONFIG["width"],
        height=PLOT_CONFIG["height"],
        template=PLOT_CONFIG["template"],
        showlegend=False,
    )
    
    # Add percentages as annotations
    total = completed + pruned + failed
    fig.add_annotation(
        x="Completed",
        y=completed,
        text=f"{completed/total*100:.1f}%",
        showarrow=False,
        yshift=20,
    )
    fig.add_annotation(
        x="Pruned",
        y=pruned,
        text=f"{pruned/total*100:.1f}%",
        showarrow=False,
        yshift=20,
    )
    
    # Save
    output_path = output_dir / "07_trial_summary.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved to: {output_path}")
    
    return fig


def export_best_params(study: optuna.Study, output_dir: Path):
    """
    Export best parameters to JSON file
    """
    print("\n📝 Exporting best parameters...")
    
    best_params = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }
    
    output_path = output_dir / "best_params.json"
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=4)
    
    print(f"✅ Saved to: {output_path}")
    print("\n" + "="*60)
    print("BEST PARAMETERS:")
    print("="*60)
    print(json.dumps(best_params, indent=2))
    print("="*60)


def export_trials_dataframe(study: optuna.Study, output_dir: Path):
    """
    Export all trials to CSV for further analysis
    """
    print("\n📊 Exporting trials dataframe...")
    
    df = study.trials_dataframe()
    
    # Save to CSV
    output_path = output_dir / "trials_dataframe.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATAFRAME SUMMARY:")
    print("="*60)
    print(df.describe())
    print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to generate all visualizations"""
    
    print("\n" + "="*60)
    print("OPTUNA VISUALIZATION SCRIPT")
    print("="*60)
    
    # Load study
    study = load_study(STUDY_NAME, STORAGE_PATH)
    
    # Generate all plots
    plot_1_optimization_history(study, OUTPUT_DIR)
    plot_2_parallel_coordinate(study, OUTPUT_DIR)
    plot_3_param_importances(study, OUTPUT_DIR)
    plot_4_contour(study, OUTPUT_DIR)
    plot_5_slice(study, OUTPUT_DIR)
    plot_6_edf(study, OUTPUT_DIR)
    plot_7_custom_summary(study, OUTPUT_DIR)
    
    # Export data
    export_best_params(study, OUTPUT_DIR)
    export_trials_dataframe(study, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()