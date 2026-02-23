"""CREME-style CRE experiment framework for AlphaGenome.

Translates the conceptual framework from CREME-NN (Koo et al., Nature Genetics 2024)
into AlphaGenome API calls: systematic tile-based perturbations for necessity testing,
higher-order interaction discovery, and CRISPRi tiling scans.

CREME works with local models on one-hot sequences. AlphaGenome is a remote API.
We map CREME's shuffle/ablation perturbations into AlphaGenome's deletion variant
mechanics (ref='N'*width, alt='N' deletions).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NecessityResult:
    """Result of a tile-based necessity test.

    Each tile in the region is deleted (replaced with N) and the effect on
    target gene expression is scored via GeneMaskLFCScorer.

    Attributes:
        scores: Tidy DataFrame from variant_scorers.tidy_scores() with all
            per-tile, per-gene, per-biosample scores.
        tile_effects: Per-tile summary DataFrame filtered to the target gene,
            with columns: tile_label, chrom, start, end, raw_score,
            quantile_score, ontology_curie.
        tiles: The list of genome.Interval tiles that were tested.
        target_gene: Gene symbol that was tested.
        context_interval: The model context interval used.
    """

    scores: pd.DataFrame
    tile_effects: pd.DataFrame
    tiles: list[genome.Interval]
    target_gene: str
    context_interval: genome.Interval

    # -- visualizations --

    def plot_bar(self, ontology_curie: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        """Horizontal bar chart of tiles ranked by LFC for the target gene.

        Args:
            ontology_curie: Filter to a specific tissue/cell type. If None,
                uses the first ontology_curie in tile_effects.
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        df = self.tile_effects.copy()
        if ontology_curie is not None:
            df = df[df['ontology_curie'] == ontology_curie]
        else:
            curie = df['ontology_curie'].iloc[0]
            df = df[df['ontology_curie'] == curie]
            ontology_curie = curie

        df = df.sort_values('raw_score', key=abs, ascending=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(df))))

        colors = ['#2166ac' if v < 0 else '#b2182b' for v in df['raw_score']]
        ax.barh(df['tile_label'], df['raw_score'], color=colors, edgecolor='none')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('LFC (log2 ALT/REF)')
        ax.set_title(f'Necessity test: {self.target_gene} | {ontology_curie}')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        return ax

    def plot_heatmap(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Heatmap of tiles x ontology terms colored by LFC.

        Args:
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        df = self.tile_effects.copy()
        pivot = df.pivot_table(
            index='tile_label', columns='ontology_curie',
            values='raw_score', aggfunc='first',
        )
        # Order rows by genomic position
        tile_order = [f'Tile {i+1}' for i in range(len(self.tiles))]
        pivot = pivot.reindex([t for t in tile_order if t in pivot.index])

        if ax is None:
            fig_h = max(3, 0.5 * len(pivot))
            fig_w = max(5, 0.8 * len(pivot.columns) + 2)
            _, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)
        cmap = plt.cm.RdBu_r
        im = ax.imshow(
            pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax,
            aspect='auto', interpolation='nearest',
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(f'Necessity: {self.target_gene} (LFC)')
        plt.colorbar(im, ax=ax, label='LFC', shrink=0.8)
        plt.tight_layout()
        return ax

    def plot_genome(
        self,
        transcript_extractor: transcript_utils.TranscriptExtractor | None = None,
        ontology_curie: str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Genome browser view with colored tile rectangles + transcripts.

        Args:
            transcript_extractor: For drawing gene models. If None, skips.
            ontology_curie: Filter scores to this tissue. Uses first if None.
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        df = self.tile_effects.copy()
        if ontology_curie is not None:
            df = df[df['ontology_curie'] == ontology_curie]
        else:
            curie = df['ontology_curie'].iloc[0]
            df = df[df['ontology_curie'] == curie]
            ontology_curie = curie

        # Determine plot region from tiles
        region_start = min(t.start for t in self.tiles)
        region_end = max(t.end for t in self.tiles)
        pad = int((region_end - region_start) * 0.1)
        view_interval = genome.Interval(
            self.tiles[0].chromosome,
            region_start - pad,
            region_end + pad,
        )

        # Build components for plot_components.plot
        components = []

        # Transcript annotation
        if transcript_extractor is not None:
            transcripts = transcript_extractor.extract(view_interval)
            components.append(plot_components.TranscriptAnnotation(transcripts))

        # We'll use matplotlib directly for tile rectangles
        if ax is None:
            n_panels = 1 + (1 if transcript_extractor else 0)
            fig, axes = plt.subplots(
                n_panels, 1, figsize=(14, 1.5 * n_panels),
                gridspec_kw={'height_ratios': [1] * n_panels},
                sharex=True,
            )
            if n_panels == 1:
                axes = [axes]
        else:
            axes = [ax]

        # If we have transcript_extractor, use plot_components for the top panel
        if transcript_extractor is not None:
            transcripts = transcript_extractor.extract(view_interval)
            plot_components.plot(
                [plot_components.TranscriptAnnotation(transcripts)],
                interval=view_interval,
                title=f'Necessity: {self.target_gene} tiles',
            )
            plt.show()

        # Draw tile rectangles on a separate figure
        fig_tiles, ax_tiles = plt.subplots(figsize=(14, 2))

        scores_by_tile = {}
        for _, row in df.iterrows():
            scores_by_tile[row['tile_label']] = row['raw_score']

        vmax = max(abs(v) for v in scores_by_tile.values()) if scores_by_tile else 1e-6
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.cm.RdBu_r

        for i, tile in enumerate(self.tiles):
            label = f'Tile {i+1}'
            score = scores_by_tile.get(label, 0.0)
            color = cmap(norm(score))
            rect = mpatches.FancyBboxPatch(
                (tile.start, 0.1), tile.width, 0.8,
                boxstyle='round,pad=0', facecolor=color,
                edgecolor='black', linewidth=0.5,
            )
            ax_tiles.add_patch(rect)
            ax_tiles.text(
                tile.start + tile.width / 2, 0.5, label,
                ha='center', va='center', fontsize=7,
            )

        ax_tiles.set_xlim(view_interval.start, view_interval.end)
        ax_tiles.set_ylim(0, 1)
        ax_tiles.set_xlabel(f'{self.tiles[0].chromosome} position')
        ax_tiles.set_yticks([])
        ax_tiles.set_title(f'Tile necessity scores ({ontology_curie})')
        ax_tiles.spines[['top', 'right', 'left']].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_tiles, label='LFC', shrink=0.6, pad=0.02)
        plt.tight_layout()
        return ax_tiles


@dataclasses.dataclass
class InteractionResult:
    """Result of a greedy multi-tile ablation (higher-order interaction test).

    At each round, the tile with the largest additional effect is removed,
    and the cumulative expression change is tracked.

    Attributes:
        rounds: List of dicts with keys:
            - round: int (1-indexed)
            - tile_removed: str (tile label)
            - tile_interval: genome.Interval
            - delta: float (additional effect from removing this tile)
            - cumulative_effect: float (total effect so far)
            - remaining_scores: pd.DataFrame (necessity scores for remaining tiles)
        target_gene: Gene symbol.
        ontology_curie: Tissue used for scoring.
    """

    rounds: list[dict[str, Any]]
    target_gene: str
    ontology_curie: str

    def plot_waterfall(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Cumulative expression change as tiles are removed.

        Step plot showing how expression drops as tiles are ablated,
        labeled with which tile was removed at each step.

        Args:
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        rounds_num = [0] + [r['round'] for r in self.rounds]
        cum_effects = [0.0] + [r['cumulative_effect'] for r in self.rounds]

        ax.step(rounds_num, cum_effects, where='post', color='#2166ac',
                linewidth=2, marker='o', markersize=6)
        ax.fill_between(rounds_num, cum_effects, step='post', alpha=0.15, color='#2166ac')

        for r in self.rounds:
            ax.annotate(
                r['tile_removed'],
                xy=(r['round'], r['cumulative_effect']),
                xytext=(5, 10), textcoords='offset points',
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
            )

        ax.set_xlabel('Tiles removed')
        ax.set_ylabel('Cumulative LFC from WT')
        ax.set_title(f'Interaction test: {self.target_gene} | {self.ontology_curie}')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        return ax

    def plot_contribution(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Per-round delta bar chart.

        Args:
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        labels = [r['tile_removed'] for r in self.rounds]
        deltas = [r['delta'] for r in self.rounds]
        colors = ['#2166ac' if d < 0 else '#b2182b' for d in deltas]

        ax.bar(labels, deltas, color=colors, edgecolor='none')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Tile removed (in order)')
        ax.set_ylabel('Delta LFC')
        ax.set_title(f'Per-round contribution: {self.target_gene}')
        ax.spines[['top', 'right']].set_visible(False)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return ax


@dataclasses.dataclass
class CRISPRiResult:
    """Result of a CRISPRi tiling scan with full track-level outputs.

    Attributes:
        tile_outputs: List of (tile_interval, VariantOutput) pairs from
            predict_variant for each tile deletion.
        tile_scores: Tidy DataFrame of scalar scores across all tiles.
        tiles: List of genome.Interval tiles tested.
        target_gene: Gene symbol.
        context_interval: Model context interval used.
    """

    tile_outputs: list[tuple[genome.Interval, Any]]
    tile_scores: pd.DataFrame
    tiles: list[genome.Interval]
    target_gene: str
    context_interval: genome.Interval

    def plot_tracks(
        self,
        tile_idx: int,
        modality: str = 'rna_seq',
        transcript_extractor: transcript_utils.TranscriptExtractor | None = None,
        zoom_width: int = 2**15,
    ) -> None:
        """Overlaid WT/CRISPRi tracks for one tile deletion.

        Args:
            tile_idx: Index into tiles/tile_outputs to plot.
            modality: Which modality to display (e.g. 'rna_seq', 'dnase').
            transcript_extractor: For gene annotation track. Optional.
            zoom_width: Width to zoom the view to.
        """
        tile, voutput = self.tile_outputs[tile_idx]

        ref_tdata = getattr(voutput.reference, modality)
        alt_tdata = getattr(voutput.alternate, modality)

        components = []
        if transcript_extractor is not None:
            transcripts = transcript_extractor.extract(self.context_interval)
            components.append(plot_components.TranscriptAnnotation(transcripts))

        components.append(
            plot_components.OverlaidTracks(
                tdata={'WT': ref_tdata, 'CRISPRi': alt_tdata},
                colors={'WT': 'dimgrey', 'CRISPRi': 'red'},
            )
        )

        variant = _make_deletion_variant(tile)
        view_iv = ref_tdata.interval.resize(zoom_width) if zoom_width else ref_tdata.interval

        plot_components.plot(
            components,
            interval=view_iv,
            annotations=[plot_components.VariantAnnotation([variant])],
            title=f'CRISPRi Tile {tile_idx+1}: {tile} | {modality}',
        )
        plt.show()

    def plot_summary(
        self,
        ontology_curie: str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Summary bar of expression change per tile.

        Args:
            ontology_curie: Filter to specific tissue. Uses first if None.
            ax: Matplotlib axes to plot on. Created if None.

        Returns:
            The matplotlib Axes object.
        """
        df = self.tile_scores.copy()
        df = df[df['gene_name'] == self.target_gene]

        if ontology_curie is not None:
            df = df[df['ontology_curie'] == ontology_curie]
        elif 'ontology_curie' in df.columns and len(df) > 0:
            curie = df['ontology_curie'].iloc[0]
            df = df[df['ontology_curie'] == curie]
            ontology_curie = curie

        # Use the tile_label column added during crispri_scan
        tile_labels = []
        scores = []
        for i in range(len(self.tiles)):
            label = f'Tile {i+1}'
            tile_df = df[df['tile_label'] == label]
            if len(tile_df) > 0:
                scores.append(tile_df['raw_score'].iloc[0])
            else:
                scores.append(0.0)
            tile_labels.append(label)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(tile_labels))))

        colors = ['#2166ac' if s < 0 else '#b2182b' for s in scores]
        y_pos = range(len(tile_labels))
        ax.barh(y_pos, scores, color=colors, edgecolor='none')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tile_labels)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('LFC (log2 ALT/REF)')
        ax.set_title(f'CRISPRi scan: {self.target_gene}' +
                      (f' | {ontology_curie}' if ontology_curie else ''))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        return ax


# ---------------------------------------------------------------------------
# Helper: build a deletion variant from an interval
# ---------------------------------------------------------------------------

def _make_deletion_variant(tile: genome.Interval, name: str | None = None) -> genome.Variant:
    """Create a deletion variant for a tile (ref='N'*width, alt='N')."""
    return genome.Variant(
        chromosome=tile.chromosome,
        position=tile.start + 1,  # 1-based
        reference_bases='N' * tile.width,
        alternate_bases='N',
        name=name or f'del_{tile.chromosome}:{tile.start}-{tile.end}',
    )


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class CREExperiment:
    """CREME-style cis-regulatory element experiment runner for AlphaGenome.

    Provides systematic tile-based perturbation tests:
    - Necessity test: which tiles are required for target gene expression?
    - Interaction test: greedy ablation to find higher-order CRE interactions.
    - CRISPRi scan: fine-resolution tiling with full track-level outputs.

    Args:
        model: AlphaGenome DNA model client (from dna_client.create()).
        gene_symbol: Target gene symbol (e.g. 'TAL1').
        gtf: Gene annotation DataFrame (GENCODE GTF feather).
        transcript_extractor: TranscriptExtractor for visualization.
        ontology_terms: List of ontology CURIEs for tissues/cell types
            (e.g. ['CL:0001059'] for CD34+ progenitors).
        seq_length: Sequence length for model context window.
            Default: SEQUENCE_LENGTH_1MB.
    """

    def __init__(
        self,
        model: dna_client.DnaClient,
        gene_symbol: str,
        gtf: pd.DataFrame,
        transcript_extractor: transcript_utils.TranscriptExtractor,
        ontology_terms: list[str],
        seq_length: int = dna_client.SEQUENCE_LENGTH_1MB,
    ):
        self.model = model
        self.gene_symbol = gene_symbol
        self.gtf = gtf
        self.transcript_extractor = transcript_extractor
        self.ontology_terms = ontology_terms
        self.seq_length = seq_length

        # Resolve gene interval
        self.gene_interval = gene_annotation.get_gene_interval(
            gtf, gene_symbol=gene_symbol,
        )
        self.context_interval = self.gene_interval.resize(seq_length)

    @staticmethod
    def tile_region(
        region: genome.Interval,
        tile_width: int = 5000,
        step: int | None = None,
    ) -> list[genome.Interval]:
        """Tile a genomic region into evenly spaced intervals.

        Args:
            region: The genomic region to tile.
            tile_width: Width of each tile in bp.
            step: Step size between tile starts. Defaults to tile_width
                (non-overlapping).

        Returns:
            List of genome.Interval tiles covering the region.
        """
        if step is None:
            step = tile_width

        tiles = []
        pos = region.start
        while pos < region.end:
            end = min(pos + tile_width, region.end)
            tiles.append(genome.Interval(region.chromosome, pos, end))
            pos += step
        return tiles

    def necessity_test(
        self,
        tiles: list[genome.Interval],
        scorers: list | None = None,
        target_gene: str | None = None,
    ) -> NecessityResult:
        """Run a tile-based necessity test.

        For each tile, creates a deletion variant (ref='N'*width, alt='N')
        and scores the effect on target gene expression.

        Args:
            tiles: List of genomic intervals to delete one at a time.
            scorers: Variant scorers to use. Defaults to recommended RNA_SEQ
                scorer (GeneMaskLFCScorer).
            target_gene: Gene to focus on. Defaults to self.gene_symbol.

        Returns:
            NecessityResult with scores, tile_effects, and visualization methods.
        """
        target_gene = target_gene or self.gene_symbol
        if scorers is None:
            scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]

        # Score each tile individually and tag with tile index so we
        # never need to reverse-engineer the variant_id format.
        per_tile_dfs = []

        for i, tile in enumerate(tqdm(tiles, desc='Necessity test')):
            variant = _make_deletion_variant(tile, name=f'Tile_{i+1}')
            result = self.model.score_variant(
                interval=self.context_interval,
                variant=variant,
                variant_scorers=scorers,
            )
            df_i = variant_scorers.tidy_scores([result], match_gene_strand=True)
            df_i['tile_idx'] = i
            df_i['tile_label'] = f'Tile {i+1}'
            df_i['chrom'] = tile.chromosome
            df_i['tile_start'] = tile.start
            df_i['tile_end'] = tile.end
            per_tile_dfs.append(df_i)

        scores_df = pd.concat(per_tile_dfs, ignore_index=True)

        # Build tile_effects: filter to target gene
        gene_scores = scores_df[scores_df['gene_name'] == target_gene].copy()

        return NecessityResult(
            scores=scores_df,
            tile_effects=gene_scores,
            tiles=tiles,
            target_gene=target_gene,
            context_interval=self.context_interval,
        )

    def interaction_test(
        self,
        tiles: list[genome.Interval],
        scorers: list | None = None,
        target_gene: str | None = None,
        num_rounds: int | None = None,
        ontology_curie: str | None = None,
    ) -> InteractionResult:
        """Greedy multi-tile ablation test (higher-order interaction).

        At each round:
        1. Score all remaining tiles via necessity test.
        2. Pick the tile with the largest |effect| on the target gene.
        3. Record its effect and remove it from the candidate set.
        4. Accumulate the total expression change.

        This approximates CREME's greedy interaction test under a linear
        additivity assumption (each tile's effect is scored independently).

        Args:
            tiles: List of genomic intervals to test.
            scorers: Variant scorers. Defaults to recommended RNA_SEQ.
            target_gene: Gene to focus on. Defaults to self.gene_symbol.
            num_rounds: Max ablation rounds. Defaults to len(tiles).
            ontology_curie: Tissue to use for picking the worst tile.
                Defaults to first available.

        Returns:
            InteractionResult with per-round ablation data and plots.
        """
        target_gene = target_gene or self.gene_symbol
        if scorers is None:
            scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]
        if num_rounds is None:
            num_rounds = len(tiles)
        num_rounds = min(num_rounds, len(tiles))

        remaining = list(range(len(tiles)))  # indices into tiles
        rounds = []
        cumulative_effect = 0.0

        for round_num in range(1, num_rounds + 1):
            # Score remaining tiles
            round_tiles = [tiles[idx] for idx in remaining]
            nec = self.necessity_test(
                round_tiles, scorers=scorers, target_gene=target_gene,
            )

            # Get per-tile effects for the target gene
            effects = nec.tile_effects
            if ontology_curie is not None:
                effects = effects[effects['ontology_curie'] == ontology_curie]
            elif len(effects) > 0:
                ontology_curie = effects['ontology_curie'].iloc[0]

            if len(effects) == 0:
                break

            # Pick tile with largest absolute effect
            worst_idx = effects['raw_score'].abs().idxmax()
            worst_row = effects.loc[worst_idx]
            worst_tile_label = worst_row['tile_label']
            worst_delta = worst_row['raw_score']

            # Map back to remaining index
            tile_num = int(worst_tile_label.split()[-1]) - 1  # 0-indexed in round_tiles
            original_idx = remaining[tile_num]

            cumulative_effect += worst_delta

            rounds.append({
                'round': round_num,
                'tile_removed': f'Tile {original_idx + 1}',
                'tile_interval': tiles[original_idx],
                'delta': float(worst_delta),
                'cumulative_effect': float(cumulative_effect),
                'remaining_scores': effects,
            })

            remaining.pop(tile_num)
            if not remaining:
                break

        return InteractionResult(
            rounds=rounds,
            target_gene=target_gene,
            ontology_curie=ontology_curie or '',
        )

    def crispri_scan(
        self,
        tiles: list[genome.Interval],
        outputs: list[dna_client.OutputType] | None = None,
        target_gene: str | None = None,
    ) -> CRISPRiResult:
        """CRISPRi tiling scan with track-level REF vs ALT comparisons.

        For each tile:
        1. Creates a deletion variant.
        2. Calls predict_variant for full REF/ALT TrackData.
        3. Calls score_variant for scalar summary.

        Args:
            tiles: List of genomic intervals to scan.
            outputs: Requested output modalities. Defaults to RNA_SEQ.
            target_gene: Gene of interest. Defaults to self.gene_symbol.

        Returns:
            CRISPRiResult with track outputs and summary scores.
        """
        target_gene = target_gene or self.gene_symbol
        if outputs is None:
            outputs = [dna_client.OutputType.RNA_SEQ]

        tile_outputs = []
        per_tile_dfs = []
        scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]

        for i, tile in enumerate(tqdm(tiles, desc='CRISPRi scan')):
            variant = _make_deletion_variant(tile, name=f'CRISPRi_Tile_{i+1}')

            # Track-level prediction
            voutput = self.model.predict_variant(
                interval=self.context_interval,
                variant=variant,
                requested_outputs=outputs,
                ontology_terms=self.ontology_terms,
            )
            tile_outputs.append((tile, voutput))

            # Scalar scoring
            score_result = self.model.score_variant(
                interval=self.context_interval,
                variant=variant,
                variant_scorers=scorers,
            )
            df_i = variant_scorers.tidy_scores(
                [score_result], match_gene_strand=True,
            )
            df_i['tile_idx'] = i
            df_i['tile_label'] = f'Tile {i+1}'
            per_tile_dfs.append(df_i)

        tile_scores = pd.concat(per_tile_dfs, ignore_index=True)

        return CRISPRiResult(
            tile_outputs=tile_outputs,
            tile_scores=tile_scores,
            tiles=tiles,
            target_gene=target_gene,
            context_interval=self.context_interval,
        )
