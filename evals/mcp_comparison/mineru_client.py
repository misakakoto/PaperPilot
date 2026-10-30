"""Invoke the unmodified official server, never a locally relabelled tool.

Run with the separate MCP environment. No project configuration or API keys load.
Transport failures remain unresolved and are never automatically resubmitted.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def clean_env(environ):
    keep = {'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE',
            'APPDATA', 'LOCALAPPDATA', 'PROGRAMFILES', 'COMSPEC'}
    result = {k:v for k,v in environ.items() if k.upper() in keep}
    result.update(PYTHON_DOTENV_DISABLED='1', PYTHONIOENCODING='utf-8', MINERU_API_TOKEN='')
    return result


def contained_markdown(entry, output_dir):
    """Full on-disk content takes precedence over truncated inline preview."""
    if entry.get('extract_path'):
        path = Path(entry['extract_path']).resolve()
        if not path.is_relative_to(Path(output_dir).resolve()) or path.suffix != '.md':
            raise ValueError('Untrusted server output path')
        return path.read_text(encoding='utf-8')
    return entry.get('content', '')


async def execute(folder, manifest, *, control=False):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    folder.mkdir(parents=True, exist_ok=True)
    out = folder/'mineru'
    out.mkdir(exist_ok=True)
    env = clean_env(os.environ)
    env['OUTPUT_DIR'] = str(out)
    transport = StdioTransport(command=sys.executable,
        args=['-c', 'from mineru_open_mcp.cli import main; main()',
              '--transport', 'stdio', '--output-dir', str(out)],
        env=env, cwd=str(folder), keep_alive=False, log_file=folder/'mcp-stderr.log')
    async with Client(transport, timeout=600, init_timeout=60) as client:
        if not (folder/'tools.json').exists():
            save(folder/'tools.json', [t.model_dump(mode='json') for t in await client.list_tools()])
        for source in manifest:
            ident = source['name'] + ('-page17' if control else '')
            receipt = folder/(ident+'-mcp.json')
            if receipt.exists():
                continue
            start = folder/(ident+'-started.json')
            if start.exists():
                raise RuntimeError('Unresolved prior MCP request; not resending')
            if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('Paper identity changed')
            file_source = {'source': source['path']}
            if control:
                file_source['pages'] = '17'
            args = {'file_sources':[file_source], 'language':'en', 'output_dir':str(out)}
            save(start, {'tool':'parse_documents', 'arguments':args, 'source_sha256':source['sha256']})
            then = time.monotonic()
            try:
                result = await client.call_tool('parse_documents', args, raise_on_error=False)
                data = result.data
                if not isinstance(data, dict):
                    data = {'raw_content':[c.model_dump(mode='json') for c in result.content]}
                save(receipt, {'elapsed_seconds':time.monotonic()-then,
                     'is_error':result.is_error, 'data':data})
                entries = data.get('results', [])
                if data.get('status') != 'success' or not entries:
                    print(json.dumps({'case':ident,'status':'SERVER_ERROR','elapsed_seconds':time.monotonic()-then}),flush=True)
                    # A definite service error is terminal for this chain.
                    break
                markdown = contained_markdown(entries[0],out)
                (folder/(ident+'.md')).write_text(markdown,encoding='utf-8')
                print(json.dumps({'case':ident,'status':data['status'], 'chars':len(markdown),
                                  'elapsed_seconds':time.monotonic()-then}),flush=True)
            except Exception as exc:
                if not receipt.exists():
                    save(folder/(ident+'-unknown.json'), {'error_type':type(exc).__name__,
                          'elapsed_seconds':time.monotonic()-then, 'state':'UNKNOWN'})
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--control',action='store_true')
    opt = parser.parse_args()
    manifest = json.loads(opt.manifest.read_text(encoding='utf-8'))
    asyncio.run(execute(opt.folder.resolve(), manifest[:1] if opt.control else manifest, control=opt.control))
