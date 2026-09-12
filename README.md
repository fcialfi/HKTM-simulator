# HKTM-simulator

Generatore di segnale RF di test modulato secondo catena CCSDS (baseline
QPSK), da iniettare via **RF-Catcher (TestTree) Capture & Playback
Application**.

Riferimenti: AWS-OSE-ICD-0063 (Arctic Weather Satellite downlink),
CCSDS 131.0-B-2 (TM Synchronization and Channel Coding), ECSS-E-ST-50-01C.

## Struttura del progetto

```
app.py                  GUI (Streamlit) con spettro/costellazione real-time
generate_signal.py      CLI, genera output_iq.raw + metadata
verify_spectrum.py      CLI, verifica banda occupata di un file IQ esistente
ccsds_chain/
  pipeline.py            orchestrazione della catena (usata da app.py e generate_signal.py)
  reed_solomon.py         RS(255,223) + interleaving
  convolutional.py        convoluzionale K=7 rate 1/2
  scrambler.py             LFSR CCSDS
  mapping.py               NRZ-L + QPSK Gray
  pulse_shaping.py         filtro RRC
  spectrum.py              PSD/banda occupata (usato da app.py e verify_spectrum.py)
  utils.py                 bit/byte helpers, I/O file IQ
```

## Parametri baseline

| Parametro       | Valore baseline          | Selezionabile |
|-----------------|---------------------------|---------------|
| Modulazione     | QPSK                       | si (solo QPSK implementata) |
| Bit rate        | 3 570 kbps (post-codifica, header incluso) | si |
| Symbol rate     | 1 785 kS/s                 | si |
| Encoding        | NRZ-L                      | si (solo NRZ-L implementata) |
| FEC RS          | RS(255,223), interleave depth 5 | si (on/off) |
| FEC convoluzionale | rate 1/2, K=7, G1=171o, G2=133o | si (on/off, invert G2) |
| Scrambler       | CCSDS, LFSR seed 0xFF      | si (on/off, default off) |
| CADU            | 4 byte ASM (0x1ACFFC1D, non codificato) + 1275 byte RS-encoded = 1279 byte | - |

## Catena implementata

```
payload -> RS(255,223) interleave x5 -> + ASM -> conv. K=7 r=1/2
        -> NRZ-L -> [scrambler] -> QPSK (Gray) -> RRC -> IQ int16 (RF-Catcher)
```

1. **Payload**: dati pseudo-casuali riproducibili (seed) per CADU, oppure
   Transfer Frame reali da file (`--payload-source`).
2. **RS(255,223)** con interleaving a profondita' 5 (byte `i` va nel
   sotto-stream `i mod 5`); libreria `reedsolo`.
3. **ASM** (4 byte) prepeso in chiaro a ogni CADU.
4. **Convoluzionale** K=7 rate 1/2, polinomi CCSDS 171/133 ottale, applicato
   in modo continuo sull'intero flusso di bit concatenato (ASM incluso; vedi
   Limitazioni), con stato del registro azzerato una sola volta all'inizio
   dell'intero segnale (non per-CADU).
5. **NRZ-L**: mapping bipolare diretto (bit 1 -> +1, bit 0 -> -1), nessuna
   codifica differenziale.
6. **Scrambler CCSDS** (opzionale): LFSR polinomio 1+x^3+x^5+x^7+x^8, seed
   0xFF, applicato per moltiplicazione nel dominio bipolare (equivalente a
   uno XOR bit-a-bit prima del mapping NRZ-L).
7. **QPSK Gray**: coppie di campioni bipolari -> I/Q, normalizzati a
   energia media unitaria per simbolo.
8. **Pulse shaping RRC** (alpha configurabile, default 0.35).
9. **Output**: file raw IQ nel formato RF-Catcher (TestTree) Capture &
   Playback -- nessun header, non compresso, non cifrato, little-endian,
   int16 a 12 bit significativi in complemento a 2 (range [-2048, 2047]),
   interleaved `I0,Q0,I1,Q1,...` -- con file `.meta.json` affiancato
   contenente i parametri usati.

## Uso

```bash
pip install -r requirements.txt
```

### GUI (consigliata)

```bash
streamlit run app.py
```

Apre un'interfaccia grafica (dark theme) con tutti i parametri della catena
nella sidebar. **Ogni modifica a un parametro ricalcola la catena e
aggiorna in tempo reale**: spettro (PSD con banda -3dB/-20dB evidenziata),
costellazione QPSK, estratto I/Q nel tempo, e le metriche (banda occupata,
durata segnale, tempo di calcolo). Include un indicatore visivo degli
stadi della pipeline attivi/disattivi ed export diretto del file IQ +
metadata dal browser (pulsanti "Scarica").

`app.py` e `generate_signal.py` condividono la stessa implementazione della
catena (`ccsds_chain/pipeline.py`): non c'e' rischio che GUI e CLI si
disallineino.

### CLI

```bash
python generate_signal.py                       # parametri baseline
python generate_signal.py --n-cadu 500 --scramble -o test.raw

python verify_spectrum.py output_iq.raw --plot spectrum.png
```

I parametri baseline sono costanti in testa a `generate_signal.py`; le
opzioni piu' comuni sono anche esposte via CLI (`--help`).

## Limitazioni

- **Rappresentazione RS Alpha/Beta**: la libreria `reedsolo` usa i parametri
  GF(256) di default (prim=0x11d, fcr=0, generator=2), cioe' una
  rappresentazione RS "convenzionale" generica. **Non e' verificato** che
  questa coincida con la rappresentazione "conventional (Beta)" CCSDS, ed e'
  comunque diversa dalla rappresentazione "dual-basis (Alpha)" richiesta da
  alcuni sistemi (es. parametro "RS Alpha" visto nel tool TestTree). Se il
  ricevitore richiede Alpha, serve una trasformazione di base GF(256)
  aggiuntiva non ancora implementata. I parametri RS (`RS_FCR`, `RS_PRIM`,
  `RS_GENERATOR` in `ccsds_chain/reed_solomon.py`) sono isolati per poter
  essere corretti una volta confermati i requisiti del ricevitore.
- **ASM e stadio convoluzionale/scrambler**: l'implementazione segue
  l'ordine letterale richiesto (RS -> +ASM -> convoluzionale -> NRZ-L ->
  scrambler -> QPSK), quindi l'ASM viene codificato convoluzionalmente
  insieme al resto del CADU, e (se lo scrambler e' attivo) e' incluso anche
  nello scrambling. In molte implementazioni CCSDS reali l'ASM resta
  **non randomizzato** (per permettere la sincronizzazione diretta sul
  pattern noto) e la codifica convoluzionale dell'ASM e' un punto che varia
  tra sistemi. Questa e' una semplificazione/assunzione da validare (vedi
  TODO).
- **Convenzione scrambler**: il LFSR CCSDS e' implementato come Fibonacci
  LFSR standard, ma l'esatta convenzione di bit-order/tap di uscita non e'
  stata verificata contro la sequenza di riferimento CCSDS 131.0-B-2.
- **Convenzione NRZ-L**: bit 1 -> +1, bit 0 -> -1; da verificare contro la
  polarita' attesa dal ricevitore/tool.
- **Payload**: attualmente dati pseudo-casuali di test (o Transfer Frame
  grezzi da file); non viene costruito un vero header di Transfer Frame
  CCSDS (VCID, contatori, CRC, ecc.).
- **Transitorio filtro RRC**: essendo applicato un solo filtro RRC (non una
  coppia Tx/Rx accoppiata), l'uscita presenta un transitorio di
  `RRC_SPAN/2` simboli in testa e in coda.

## TODO

- [ ] Confermare formato input IQ Converter (contattare support@test-tree.com
      se necessario)
- [ ] Verificare rappresentazione RS Alpha/Beta richiesta dal ricevitore
- [ ] Validare scrambling (se applicato su ASM+dati o solo dati)
- [ ] Validare inclusione/esclusione dell'ASM nella codifica convoluzionale
- [ ] Verificare convenzione bit-order del LFSR scrambler contro la
      sequenza di riferimento CCSDS 131.0-B-2
- [ ] Verificare polarita' NRZ-L attesa
- [ ] Test end-to-end: generazione -> IQ Converter -> Capture & Playback ->
      loopback RX

## Verifica spettro

`verify_spectrum.py` calcola la PSD (media di periodogrammi con finestra di
Hanning) e misura la banda occupata a -3dB e -20dB attorno al centro banda.
Con i parametri baseline (Rs=1.785 MS/s, alpha=0.35) ci si attende una banda
occupata a -3dB vicina a Rs (~1.7-1.8 MHz) e a -20dB vicina a
Rs*(1+alpha) (~2.3-2.4 MHz).
