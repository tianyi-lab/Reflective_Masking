from __future__ import annotations

import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer


@dataclass
class BaseVisualizer(ABC):
    tokenizer: PreTrainedTokenizer

    @abstractmethod
    def visualize(self, history: list[torch.Tensor | list], **kwargs):
        raise NotImplementedError


@dataclass
class VideoVisualizer(BaseVisualizer):

    def visualize(
        self,
        history: list[torch.Tensor | list],
        output_path: str = "visualization.gif",
        **kwargs,
    ):
        raise NotImplementedError


@dataclass
class TerminalVisualizer(BaseVisualizer):

    HEADER_SIZE = 3
    PROGRESS_SIZE = 3
    PANEL_PADDING_TOP = 1
    PANEL_PADDING_BOTTOM = 1
    PANEL_PADDING_SIDE = 1
    PANEL_BORDER = 2
    MIN_TOTAL_HEIGHT = 10
    MAX_TOTAL_HEIGHT = 60
    DEFAULT_TERM_WIDTH = 120
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    def visualize(
        self,
        history: list[torch.Tensor],
        fps: int = 16,
        rich: bool = True,
        title: str = "dllm",
        max_chars: int = None,
        every_n_steps: int = 1,
        show_header: bool = True,
        skip_special_tokens: bool = False,
    ) -> None:
        try:
            first_step = history[0]
            if first_step.dim() > 1 and first_step.shape[0] > 1:
                B = first_step.shape[0]
                for b_idx in range(B):
                    seq_history = [step[b_idx].unsqueeze(0) for step in history]
                    self.visualize_one_history(
                        seq_history,
                        fps,
                        rich,
                        title=f"{title} (Batch {b_idx})",
                        max_chars=max_chars,
                        every_n_steps=every_n_steps,
                        show_header=show_header,
                        skip_special_tokens=skip_special_tokens,
                    )
            else:
                self.visualize_one_history(
                    history,
                    fps,
                    rich,
                    title,
                    max_chars,
                    every_n_steps,
                    show_header,
                    skip_special_tokens,
                )
        except Exception as e:
            print(f"(Visualization skipped due to error: {e})")

    def visualize_one_history(
        self,
        history: list[torch.Tensor],
        fps: int = 16,
        rich: bool = True,
        title: str = "dllm",
        max_chars: int = None,
        every_n_steps: int = 1,
        show_header: bool = True,
        skip_special_tokens: bool = False,
    ) -> None:
        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.live import Live
            from rich.panel import Panel
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeRemainingColumn,
            )
            from rich.text import Text

            _RICH_IMPORTED = True
        except Exception:
            _RICH_IMPORTED = False

        try:
            from tqdm import tqdm

            _TQDM_IMPORTED = True
        except Exception:
            _TQDM_IMPORTED = False

        if self.tokenizer is None:
            raise ValueError(
                "TerminalVisualizer.tokenizer must be set to a valid tokenizer."
            )

        tokenizer = self.tokenizer
        specials: set[int] = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._specials = specials
        self._mask_token_id: Optional[int] = getattr(tokenizer, "mask_token_id", None)
        self._pad_token_id: Optional[int] = getattr(tokenizer, "pad_token_id", None)
        self._eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)


        sleep_s = 0.0 if fps <= 0 else 1.0 / float(max(1, fps))
        total_steps = len(history)
        every_n_steps = max(1, int(every_n_steps))

        final_text = self._detok(history[-1], skip_special_tokens=skip_special_tokens)
        final_text = self._truncate(final_text, max_chars)

        import shutil
        import textwrap

        def strip_ansi(s: str) -> str:
            return self.ansi_escape.sub("", s) if s else ""

        def estimate_height_from_text(text: str, console_width: int) -> int:
            plain = strip_ansi(text or "")
            inner_width = max(
                10, console_width - 2 * self.PANEL_PADDING_SIDE - self.PANEL_BORDER
            )
            lines = 0
            for para in plain.splitlines() or [""]:
                if para.strip() == "":
                    lines += 1
                    continue
                wrapped = textwrap.wrap(
                    para,
                    width=inner_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                lines += max(1, len(wrapped))
            text_block_lines = (
                lines + self.PANEL_PADDING_TOP + self.PANEL_PADDING_BOTTOM
            )
            extra = 2
            header_h = self.HEADER_SIZE if show_header else 0
            total = header_h + text_block_lines + self.PROGRESS_SIZE + extra
            total = max(self.MIN_TOTAL_HEIGHT, min(total, self.MAX_TOTAL_HEIGHT))
            return int(total)

        try:
            term_width = shutil.get_terminal_size().columns
            if not isinstance(term_width, int) or term_width <= 0:
                term_width = self.DEFAULT_TERM_WIDTH
        except Exception:
            term_width = self.DEFAULT_TERM_WIDTH

        est_height = estimate_height_from_text(final_text, console_width=term_width)

        use_rich = bool(rich and _RICH_IMPORTED)

        if not use_rich or not _RICH_IMPORTED:
            if not _TQDM_IMPORTED:
                for i, toks in enumerate(history, start=1):
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                print("\n✨ Generation complete!\n")
                print(final_text)
                return

            pbar = tqdm(total=total_steps, desc="Diffusion", leave=True)
            for i, toks in enumerate(history, start=1):
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "masks": self._count_masks(toks),
                        "pct": f"{int(100 * i / max(total_steps, 1))}%",
                    }
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
            pbar.close()
            print("\n✨ Generation complete!\n")
            if final_text:
                print(final_text)
            return

        console = Console(
            force_terminal=True,
            color_system="truecolor",
            width=term_width,
            height=est_height,
        )
        layout = Layout()
        layout.split_column(
            (
                Layout(name="header", size=3)
                if show_header
                else Layout(name="header", size=0)
            ),
            Layout(name="text", ratio=1),
            Layout(name="progress", size=3),
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Diffusion"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]Masks: {task.fields[masks]}"),
            TextColumn("•"),
            TextColumn("[magenta]{task.fields[pct]:>4s}"),
            TimeRemainingColumn(),
            expand=True,
        )

        init_masks = self._count_masks(history[0]) if history else 0
        task_id = progress.add_task(
            "Generating", total=total_steps, masks=init_masks, pct="0%"
        )

        with Live(layout, console=console, refresh_per_second=max(1, fps)):
            for step_idx, toks in enumerate(history, start=1):
                if show_header:
                    header = Text(title, style="bold magenta", justify="center")
                    layout["header"].update(Panel(header, border_style="bright_blue"))

                masks_remaining = self._count_masks(toks)
                pct = f"{int(100 * step_idx / max(total_steps, 1))}%"
                progress.update(task_id, advance=1, masks=masks_remaining, pct=pct)

                if (
                    every_n_steps <= 1
                    or (step_idx % every_n_steps == 0)
                    or step_idx in (1, total_steps)
                ):
                    text_str = self._detok(
                        toks, skip_special_tokens=skip_special_tokens
                    )
                    text_str = self._truncate(text_str, max_chars)
                    text_rich = Text.from_ansi(text_str) if text_str else Text("")
                    layout["text"].update(
                        Panel(
                            (
                                text_rich
                                if text_rich.plain
                                else Text("[dim]— no tokens —[/dim]")
                            ),
                            title="[bold]Generated Text",
                            subtitle=f"[dim]Step {step_idx}/{total_steps}[/dim]",
                            border_style="cyan",
                            padding=(1, 1),
                        )
                    )

                layout["progress"].update(Panel(progress))
                if sleep_s > 0:
                    time.sleep(sleep_s)

        console.print("\n[bold green]✨ Generation complete![/bold green]\n")


    def _has_tty(self) -> bool:
        return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")

    def _first_item(self, x: torch.Tensor) -> torch.Tensor:
        return x[0] if x.dim() > 1 else x

    def _count_masks(self, toks: torch.Tensor) -> int:
        if getattr(self, "_mask_token_id", None) is None:
            return 0
        t = self._first_item(toks)
        return int((t == self._mask_token_id).sum().item())

    def _detok(self, ids_or_tensor, *, skip_special_tokens: bool) -> str:
        tokenizer = self.tokenizer
        if isinstance(ids_or_tensor, torch.Tensor):
            t = self._first_item(ids_or_tensor).long()
            ids = t.tolist()
        elif isinstance(ids_or_tensor, (list, tuple)):
            ids = list(ids_or_tensor)
        else:
            return ""

        if skip_special_tokens:
            keep = []
            specials = getattr(self, "_specials", set())
            pad_id = getattr(self, "_pad_token_id", None)
            eos_id = getattr(self, "_eos_token_id", None)
            for tid in ids:
                if tid in specials:
                    continue
                if pad_id is not None and tid == pad_id:
                    continue
                if eos_id is not None and tid == eos_id:
                    continue
                keep.append(tid)
            ids = keep

        text = ""
        try:
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            else:
                toks = tokenizer.convert_ids_to_tokens(ids)
                if hasattr(tokenizer, "convert_tokens_to_string"):
                    text = tokenizer.convert_tokens_to_string(toks)
                else:
                    text = " ".join(map(str, toks))
        except Exception:
            try:
                text = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                text = ""

        if text:
            text = text.replace("\r", "")
        return text

    def _truncate(self, s: str, max_chars: Optional[int]) -> str:
        if max_chars is None or (isinstance(max_chars, int) and max_chars < 0):
            return s
        return s[:max_chars]


if __name__ == "__main__":
    pass
