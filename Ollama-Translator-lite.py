# -*- coding: utf-8 -*-
#    MTUOC-Translator-Ollama
#    Copyright (C) 2026  Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import codecs
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import requests


DEFAULT_PROMPT = "Translate the following text from {sl} into {tl}.\n{sl}: {text} \n{tl}:"


def print_info(message1, message2=""):
    print(message1, message2)


def load_prompt_template(prompt_file, prompt_arg, encoding="utf-8"):
    """Carrega la plantilla del prompt des d'un fitxer o fa servir el valor de línia de comandes/defecte."""
    if prompt_file:
        if not os.path.exists(prompt_file):
            print_info("Error: Prompt file not found:", prompt_file)
            sys.exit(1)
        with codecs.open(prompt_file, "r", encoding=encoding) as pf:
            return pf.read().rstrip("\r\n")
    return prompt_arg


def connect():
    global urlOllama, sessionOllama
    try:
        urlOllama = f"http://{args.ip}:{args.port}/api/chat"
        sessionOllama = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=args.batch_size,
            pool_maxsize=args.batch_size
        )
        sessionOllama.mount("http://", adapter)
        sessionOllama.mount("https://", adapter)
    except Exception as e:
        print_info("Error connecting to Ollama:\n", str(e))


def translate_segment(segment):
    global urlOllama, sessionOllama, prompt_template
    if not segment.strip():
        return ""
    translation = ""
    try:
        prompt_text = prompt_template.format(
            sl=args.sl,
            tl=args.tl,
            text=segment
        )
        payload = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": args.temperature,
                "num_predict": args.max_tokens
            }
        }
        response = sessionOllama.post(urlOllama, json=payload, timeout=180)
        response.raise_for_status()
        target = response.json()
        translation = target["message"]["content"].strip()
        translation = translation.replace("\n", " ")
    except Exception as e:
        print_info("Error retrieving translation from Ollama:\n", str(e))
    return translation


def translate_batch(segments_batch):
    """Tradueix un lot de segments en paral·lel preservant l'ordre d'entrada."""
    if not segments_batch:
        return []
    if len(segments_batch) == 1:
        return [translate_segment(segments_batch[0])]

    with ThreadPoolExecutor(max_workers=len(segments_batch)) as executor:
        translations = list(executor.map(translate_segment, segments_batch))
    return translations


parser = argparse.ArgumentParser(
    description='MTUOC-Translator-Ollama: command line batch translator for Ollama servers.'
)
parser.add_argument('--ip', dest='ip', help='The IP of the Ollama server (e.g., localhost)', action='store', default='localhost')
parser.add_argument('--port', dest='port', help='The port used by the Ollama server (default: 11434)', action='store', default='11434')
parser.add_argument('-i', '--input', dest='input', help='The input file', action='store', required=True)
parser.add_argument('-o', '--output', dest='output', help='The output file', action='store', required=True)
parser.add_argument('-e', '--encoding', dest='encoding', help='The character encoding for input and output', action='store', default='utf-8')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=4, help='Number of segments to send concurrently per batch (default: 4)')
parser.add_argument('--model', dest='model', default='gemma3:12b', help='Model name served by Ollama (default: gemma3:12b)')
parser.add_argument('--sl', dest='sl', default='English', help='Source language name for the prompt (default: English)')
parser.add_argument('--tl', dest='tl', default='Catalan', help='Target language name for the prompt (default: Catalan)')
parser.add_argument(
    '--prompt-file',
    dest='prompt_file',
    default=None,
    help='Path to a text file containing the prompt template with {sl}, {tl}, and {text}'
)
parser.add_argument(
    '--prompt',
    dest='prompt',
    default=DEFAULT_PROMPT,
    help='Prompt template string using {sl}, {tl}, and {text} if --prompt-file is not provided'
)
parser.add_argument('--temperature', dest='temperature', type=float, default=0.0, help='Sampling temperature (default: 0.0)')
parser.add_argument('--max-tokens', dest='max_tokens', type=int, default=512, help='Maximum generated tokens per segment (default: 512)')

args = parser.parse_args()

prompt_template = load_prompt_template(args.prompt_file, args.prompt, args.encoding)
connect()

entrada = codecs.open(args.input, "r", encoding=args.encoding)
sortida = codecs.open(args.output, "w", encoding=args.encoding)

cont = 0
start_time = time.time()
chars = 0
batch_buffer = []

for linia in entrada:
    linia = linia.rstrip("\r\n")
    batch_buffer.append(linia)

    if len(batch_buffer) >= args.batch_size:
        traduccions_batch = translate_batch(batch_buffer)
        for src_seg, tgt_seg in zip(batch_buffer, traduccions_batch):
            cont += 1
            chars += len(src_seg)
            print(cont)
            print(src_seg)
            print(tgt_seg)
            print("------------------------------------------")
            sortida.write(tgt_seg + "\n")
        batch_buffer = []

if batch_buffer:
    traduccions_batch = translate_batch(batch_buffer)
    for src_seg, tgt_seg in zip(batch_buffer, traduccions_batch):
        cont += 1
        chars += len(src_seg)
        print(cont)
        print(src_seg)
        print(tgt_seg)
        print("------------------------------------------")
        sortida.write(tgt_seg + "\n")

entrada.close()
sortida.close()

end_time = time.time()
total_time = end_time - start_time

print("TOTAL TIME:", round(total_time, 2))
print("TOTAL SEGMENTS:", cont)
print("TOTAL CHARS:", chars)
ss = cont / total_time if total_time > 0 else 0
print("SEGMENTS/SECONDS:", round(ss, 2))
cs = chars / total_time if total_time > 0 else 0
print("CHARS/SECONDS:", round(cs, 2))
