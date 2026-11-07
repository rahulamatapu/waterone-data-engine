"""Command-line interface for WaterOne Data Engine ('waterpipe')."""
import os
import subprocess
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.pipeline import WaterOnePipeline

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="waterpipe")
def main():
    """💧 waterpipe: Unified CLI Suite for WaterOne Data Engine lakehouse."""
    pass


@main.command()
def status():
    """📊 Show pipeline health, layer row counts, and audit logs."""
    pipeline = WaterOnePipeline()
    stat = pipeline.status()

    if not stat.database_exists:
        console.print("[yellow]⚠️ Lakehouse database does not exist yet. Run 'waterpipe run' to initialize.[/yellow]")
        return

    # Header Panel
    size_mb = stat.database_size_bytes / (1024 * 1024)
    console.print(
        Panel(
            f"[bold cyan]WaterOne Data Engine Catalog[/bold cyan]\n"
            f"[dim]PWS ID: KS2009110 | Location: database/water_lakehouse.duckdb | Size: {size_mb:.2f} MB[/dim]",
            border_style="cyan",
            expand=False,
        )
    )

    # Tables & Layers Table
    table = Table(title="Lakehouse Layers & Tables", box=box.ROUNDED)
    table.add_column("Layer", style="bold magenta")
    table.add_column("Table / View Name", style="bold green")
    table.add_column("Record Count", justify="right", style="cyan")

    for tbl_name, count in sorted(stat.layer_counts.items()):
        if tbl_name.startswith("bronze_"):
            layer = "Bronze (Raw)"
        elif tbl_name.startswith("dim_") or tbl_name.startswith("fact_"):
            layer = "Silver (Star Schema)"
        elif tbl_name.startswith("mart_"):
            layer = "Gold (Analytics & ML)"
        else:
            layer = "Audit & Metadata"

        cnt_str = f"{count:,}" if count >= 0 else "N/A"
        table.add_row(layer, tbl_name, cnt_str)

    console.print(table)

    # Recent Audit Log Table
    if stat.recent_audits:
        audit_tbl = Table(title="Recent Ingestion Audits", box=box.SIMPLE)
        audit_tbl.add_column("File Name", style="white")
        audit_tbl.add_column("Layer", style="yellow")
        audit_tbl.add_column("Status", style="bold green")
        audit_tbl.add_column("Rows", justify="right")
        for a in stat.recent_audits:
            st_color = "green" if a["status"] == "SUCCESS" else "red"
            audit_tbl.add_row(a["file_name"], a["layer"], f"[{st_color}]{a['status']}[/{st_color}]", f"{a['row_count']:,}")
        console.print(audit_tbl)


@main.command()
@click.option("--force", is_flag=True, help="Re-ingest files even if previously processed.")
def ingest(force: bool):
    """📥 Ingest raw landing files into Bronze (CCR PDF, KDHE CSVs, USGS Telemetry)."""
    pipeline = WaterOnePipeline()
    with console.status("[bold green]Ingesting Landing files into Bronze layer..."):
        res = pipeline.ingest(force=force)

    console.print("[bold green]✅ Bronze Ingestion Complete![/bold green]")
    for c in res.get("kdhe_csvs", []):
        console.print(f"  • CSV: [white]{c['file']}[/white] -> [bold cyan]{c['status']}[/bold cyan] ({c.get('row_count', 0):,} rows)")
    for p in res.get("ccr_pdfs", []):
        console.print(f"  • PDF: [white]{p['file']}[/white] -> [bold cyan]{p['status']}[/bold cyan] ({p.get('row_count', 0):,} rows)")
    u = res.get("usgs_telemetry", {})
    console.print(f"  • USGS River Telemetry: [bold cyan]{u.get('status', 'N/A')}[/bold cyan] ({u.get('row_count', 0):,} readings)")


@main.command()
def transform():
    """⚙️ Run Silver Star Schema transformations and Gold feature marts."""
    pipeline = WaterOnePipeline()
    with console.status("[bold green]Executing Silver & Gold transformations..."):
        res = pipeline.transform()

    console.print("[bold green]✅ Transformations Complete![/bold green]")
    console.print("  [bold magenta]Silver Tables Built:[/bold magenta]")
    for tbl, count in res["silver"].items():
        console.print(f"    • {tbl}: {count:,} rows")
    console.print("  [bold yellow]Gold Marts Built:[/bold yellow]")
    for tbl, count in res["gold"].items():
        console.print(f"    • {tbl}: {count:,} rows")


@main.command()
def validate():
    """🛡️ Execute Pandera data quality gates on Bronze/Silver tables."""
    pipeline = WaterOnePipeline()
    res = pipeline.validate()
    audit_log = res.get("audit_log", [])

    if not audit_log:
        console.print("[yellow]No quality audit records found. Run 'waterpipe transform' first.[/yellow]")
        return

    table = Table(title="Pandera Quality Contracts Verification", box=box.ROUNDED)
    table.add_column("Table", style="bold green")
    table.add_column("Status", style="bold")
    table.add_column("Records Checked", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Timestamp", style="dim")

    for r in audit_log[:10]:
        status_str = "[bold green]PASSED[/bold green]" if r["is_valid"] else "[bold red]FAILED[/bold red]"
        err_color = "green" if r["error_count"] == 0 else "red"
        table.add_row(
            r["table_name"],
            status_str,
            f"{r['records_checked']:,}",
            f"[{err_color}]{r['error_count']}[/{err_color}]",
            str(r["checked_at"])[:19],
        )

    console.print(table)


@main.command()
@click.option("--contamination", type=float, default=None, help="Isolation Forest contamination rate (e.g. 0.05).")
def detect(contamination: Optional[float]):
    """🚨 Run Isolation Forest anomaly detection and pre-breach alerts."""
    pipeline = WaterOnePipeline()
    with console.status("[bold red]Training Isolation Forest & scoring river stress events..."):
        df_anom = pipeline.detect(contamination=contamination)

    crit_count = (df_anom["alert_level"] == "CRITICAL_ALERT").sum()
    warn_count = (df_anom["alert_level"] == "PRE_BREACH_WARNING").sum()

    console.print(f"[bold green]✅ Anomaly Detection Complete![/bold green] Scored {len(df_anom):,} observation days.")
    console.print(f"  • [bold red]Critical Alerts: {crit_count}[/bold red]")
    console.print(f"  • [bold yellow]Pre-Breach Warnings: {warn_count}[/bold yellow]")
    console.print(f"  • [bold green]Normal Operations: {(df_anom['alert_level'] == 'NORMAL').sum()}[/bold green]")

    flagged = df_anom[df_anom["alert_level"] != "NORMAL"].sort_values("severity_index", ascending=False).head(5)
    if not flagged.empty:
        tbl = Table(title="Top Flagged Pre-Breach Events", box=box.ROUNDED)
        tbl.add_column("Date", style="cyan")
        tbl.add_column("Severity (0-100)", justify="right", style="bold red")
        tbl.add_column("Alert Level", style="bold yellow")
        tbl.add_column("Primary Contributing Factor", style="magenta")
        for _, row in flagged.iterrows():
            tbl.add_row(
                str(row["date"])[:10],
                f"{row['severity_index']:.1f}",
                row["alert_level"],
                row["primary_contributing_factor"],
            )
        console.print(tbl)


@main.command()
@click.option("--force", is_flag=True, help="Force re-ingestion of raw landing files.")
def run(force: bool):
    """🚀 Run the complete end-to-end pipeline (Ingest -> Validate -> Transform -> Detect)."""
    console.print(Panel("[bold cyan]WaterOne Data Engine - Full Pipeline Execution[/bold cyan]", border_style="cyan"))
    pipeline = WaterOnePipeline()

    console.print("[1/3] 📥 Ingesting landing files into Bronze layer...")
    pipeline.ingest(force=force)

    console.print("[2/3] ⚙️ Building Silver Star Schema & Gold Feature Marts...")
    pipeline.transform()

    console.print("[3/3] 🚨 Training ML Anomaly Detection model...")
    df_anom = pipeline.detect()

    warns = (df_anom["alert_level"] == "PRE_BREACH_WARNING").sum()
    crits = (df_anom["alert_level"] == "CRITICAL_ALERT").sum()

    console.print(
        Panel(
            f"[bold green]✨ Pipeline Execution Successful![/bold green]\n"
            f"• Lakehouse catalog: [cyan]database/water_lakehouse.duckdb[/cyan]\n"
            f"• Scored Days: [white]{len(df_anom):,}[/white] | Pre-Breach Warnings: [yellow]{warns}[/yellow] | Critical Alerts: [red]{crits}[/red]\n"
            f"• Run [bold cyan]waterpipe status[/bold cyan] or [bold cyan]waterpipe dashboard[/bold cyan]",
            border_style="green",
        )
    )


@main.command()
@click.argument("sql")
def query(sql: str):
    """🔍 Run an ad-hoc SQL query against the local DuckDB lakehouse."""
    pipeline = WaterOnePipeline()
    try:
        df = pipeline.execute_query(sql)
        tbl = Table(title="DuckDB Query Results", box=box.ROUNDED)
        for col in df.columns:
            tbl.add_column(str(col), style="cyan")
        for _, row in df.head(50).iterrows():
            tbl.add_row(*[str(val) for val in row])
        console.print(tbl)
        console.print(f"[dim]Returned {len(df):,} rows (showing first {min(len(df), 50)})[/dim]")
    except Exception as e:
        console.print(f"[bold red]Query Error:[/bold red] {e}")


@main.command()
@click.argument("mart")
@click.argument("output_path")
@click.option("--format", "format_type", default="csv", help="Format: csv, parquet, json")
def export(mart: str, output_path: str, format_type: str):
    """📤 Export a Gold analytical mart to CSV, Parquet, or JSON."""
    pipeline = WaterOnePipeline()
    try:
        out_p = pipeline.export_mart(mart, output_path, format_type)
        console.print(f"[bold green]✅ Exported '{mart}' to {out_p} ({format_type.upper()})[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Export Error:[/bold red] {e}")


@main.command()
@click.option("--port", default=8501, help="Port to run Streamlit on.")
def dashboard(port: int):
    """🌐 Launch the interactive Streamlit Health Monitor at localhost:8501."""
    app_path = Path("dashboard/app.py")
    if not app_path.exists():
        console.print(f"[bold red]Error:[/bold red] Dashboard script not found at {app_path}")
        return

    console.print(f"[bold cyan]Launching WaterOne Health Monitor on http://localhost:{port}...[/bold cyan]")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

