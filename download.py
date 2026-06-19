from __future__ import annotations

import json
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the downloader usable without tqdm.
    tqdm = None

ROOT = Path('data/raw/public')
ROOT.mkdir(parents=True, exist_ok=True)
CHUNK = 1024 * 1024 * 4


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f'skip existing {dest} ({dest.stat().st_size} bytes)', flush=True)
        return

    part = dest.with_suffix(dest.suffix + '.part')
    start = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent': 'drone-sound-classification-dataset-fetcher/0.1'}
    if start:
        headers['Range'] = f'bytes={start}-'
    request = urllib.request.Request(url, headers=headers)
    print(f'download {url} -> {dest}', flush=True)
    if start:
        print(f'  resume from {start} bytes', flush=True)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if start and error.code == 416:
            part.rename(dest)
            print(f'  server says range complete; finalized {dest}', flush=True)
            return
        raise

    mode = 'ab' if start and response.status == 206 else 'wb'
    if start and response.status != 206:
        print('  server did not honor range; restarting file', flush=True)
        start = 0
    downloaded = start
    total_size = expected_size
    if total_size is None:
        content_length = response.headers.get('Content-Length')
        if content_length:
            try:
                total_size = start + int(content_length)
            except ValueError:
                total_size = None

    progress = _make_progress(dest.name, total_size, downloaded)
    initial_downloaded = downloaded
    started_at = time.time()
    last_report = time.time()
    try:
        with response, part.open(mode) as file:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
                else:
                    now = time.time()
                    if now - last_report >= 10:
                        _print_progress(downloaded, total_size, initial_downloaded, started_at)
                        last_report = now
    finally:
        if progress is not None:
            progress.close()

    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f'size mismatch for {dest}: got {part.stat().st_size}, expected {expected_size}')
    part.rename(dest)
    print(f'done {dest} ({dest.stat().st_size} bytes)', flush=True)


def _make_progress(name: str, total_size: int | None, initial: int):
    if tqdm is None:
        return None
    return tqdm(
        total=total_size,
        initial=initial,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )


def _print_progress(
    downloaded: int,
    total_size: int | None,
    initial_downloaded: int,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    transferred = max(downloaded - initial_downloaded, 0)
    speed = transferred / elapsed
    if total_size:
        remaining = max(total_size - downloaded, 0)
        eta = remaining / speed if speed else 0.0
        pct = downloaded / total_size * 100
        print(
            f'  {_format_bytes(downloaded)}/{_format_bytes(total_size)} '
            f'({pct:.1f}%), remaining {_format_bytes(remaining)}, '
            f'{_format_bytes(speed)}/s, eta {_format_duration(eta)}',
            flush=True,
        )
    else:
        print(
            f'  {_format_bytes(downloaded)} downloaded, '
            f'{_format_bytes(speed)}/s',
            flush=True,
        )


def _format_bytes(value: float) -> str:
    units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f'{size:.2f} {unit}'
        size /= 1024.0
    return f'{size:.2f} TiB'


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours:d}h{minutes:02d}m{secs:02d}s'
    if minutes:
        return f'{minutes:d}m{secs:02d}s'
    return f'{secs:d}s'


def extract_archive(archive: Path, target_dir: Path) -> None:
    marker = target_dir / f'.extracted-{archive.name}.done'
    if marker.exists():
        print(f'skip extracted {archive}', flush=True)
        return
    print(f'extract {archive} -> {target_dir}', flush=True)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    elif archive.name.endswith('.tar.gz') or archive.name.endswith('.tgz'):
        with tarfile.open(archive) as tf:
            tf.extractall(target_dir)
    else:
        raise RuntimeError(f'unsupported archive: {archive}')
    marker.write_text('ok\n', encoding='utf-8')
    print(f'done extract {archive}', flush=True)


def download_figshare_dronenoise() -> None:
    dataset_dir = ROOT / 'dronenoise_database'
    data = fetch_json('https://api.figshare.com/v2/articles/22133411')
    wavs = [f for f in data.get('files', []) if str(f.get('name', '"''"')).lower().endswith('.wav')]
    print(f'dronenoise_database: {len(wavs)} wav files', flush=True)
    for file_info in wavs:
        download_file(
            file_info['download_url'],
            dataset_dir / file_info['name'],
            int(file_info['size']),
        )


def download_zenodo_archives(dataset_id: str, record_id: str, suffixes: tuple[str, ...]) -> None:
    dataset_dir = ROOT / dataset_id
    archive_dir = dataset_dir / '_archives'
    data = fetch_json(f'https://zenodo.org/api/records/{record_id}')
    files = [
        f for f in data.get('files', [])
        if str(f.get('key', '"''"')).lower().endswith(suffixes)
    ]
    print(f'{dataset_id}: {len(files)} archive files', flush=True)
    for file_info in files:
        name = file_info['key']
        archive = archive_dir / name
        url = file_info.get('links', {}).get('self') or file_info.get('links', {}).get('content')
        download_file(url, archive, int(file_info['size']))
        extract_archive(archive, dataset_dir)


def download_nasa() -> None:
    dataset_id = 'nasa_small_uas_flyover'
    dataset_dir = ROOT / dataset_id
    archive = dataset_dir / '_archives' / 'small_uav_acoustics.zip'
    download_file(
        'https://data.nasa.gov/docs/datasets/rfk401li/small_uav_acoustics.zip',
        archive,
        None,
    )
    extract_archive(archive, dataset_dir)


def count_wavs() -> None:
    print('wav counts:', flush=True)
    for dataset_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        count = sum(1 for _ in dataset_dir.rglob('*.wav'))
        size = sum(p.stat().st_size for p in dataset_dir.rglob('*.wav'))
        print(f'  {dataset_dir.name}: {count} wavs, {size/1024/1024/1024:.2f} GiB wav data', flush=True)


def main() -> None:
    download_figshare_dronenoise()
    download_zenodo_archives('kielce_uav_acoustic_signals', '15190811', ('.zip',))
    download_nasa()
    download_zenodo_archives('aerosonicdb_ypad_0523', '8368882', ('.zip',))
    download_zenodo_archives('idmt_traffic', '7551553', ('.zip',))
    download_zenodo_archives('urbansound8k', '1203745', ('.tar.gz',))
    count_wavs()
    print('download/extract complete', flush=True)


if __name__ == '__main__':
    main()
