# HKTM-simulator

Generatore di segnale RF di test modulato secondo catena CCSDS (baseline
QPSK), da iniettare via **RF-Catcher (TestTree) Capture & Playback
Application**.

Riferimenti: AWS-OSE-ICD-0063 (Arctic Weather Satellite downlink),
CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, Sept. 2023),
ECSS-E-ST-50-01C.

## Struttura del progetto

```
app.py                  GUI (Streamlit) con spettro/costellazione real-time
generate_signal.py      CLI, genera output_iq.raw + metadata
verify_spectrum.py      CLI, verifica banda occupata di un file IQ esistente
ccsds_chain/
  pipeline.py            orchestrazione della catena (usata da app.py e generate_signal.py)
  reed_solomon.py         RS(255,223)/(255,239), GF(256) e dual-basis CCSDS-nativi
  convolutional.py        convoluzionale K=7, rate 1/2 punturato a 2/3-7/8
  scrambler.py             pseudo-randomizer CCSDS (131071-bit e 255-bit legacy)
  mapping.py               NRZ-L + QPSK Gray
  pulse_shaping.py         filtro RRC
  spectrum.py              PSD/banda occupata (usato da app.py e verify_spectrum.py)
  utils.py                 bit/byte helpers, I/O file IQ
```

## Parametri baseline

| Parametro       | Valore baseline          | Selezionabile |
|-----------------|---------------------------|---------------|
| Modulazione     | QPSK                       | si (solo QPSK implementata) |
| Bit rate        | 3 570 kbps (post-codifica, header incluso) | si (o symbol rate, sincronizzati) |
| Symbol rate     | 1 785 kS/s                 | si |
| Encoding        | NRZ-L                      | si (solo NRZ-L implementata) |
| FEC RS          | RS(255,223), interleave depth 5 | si (on/off; E=8 o 16; I=1,2,3,4,5,8) |
| FEC convoluzionale | rate 1/2, K=7, G1=171o, G2=133o | si (on/off; rate 1/2, 2/3, 3/4, 5/6, 7/8; invert G2 solo a rate 1/2) |
| Pseudo-randomizer | nessuno                  | si (assente / 255-bit legacy / 131071-bit, default standard dal 2023) |
| CADU            | 4 byte ASM (0x1ACFFC1D, non codificato) + 1275 byte RS-encoded = 1279 byte | dipende da I (CADU = 4 + 255*I byte) |

## Catena implementata

Segue la struttura ufficiale CCSDS 131.0-B-3 ("Overall Structure of Channel
Coding"): RS encode -> pseudo-random -> attach ASM (= CADU) -> convoluzionale
sul flusso di CADU:

```
payload -> RS(255,223) interleave x5 -> [scrambler, esclude ASM]
        -> + ASM (per CADU, forma il CADU) -> conv. K=7 r=1/2 (sul flusso di CADU)
        -> NRZ-L -> QPSK (Gray) -> RRC -> IQ int16 (RF-Catcher)
```

1. **Payload**: dati pseudo-casuali riproducibili (seed) per CADU, oppure dati
   reali da file (`--payload-source`). Il formato di questi dati e' selezionabile
   con `--input-format` (o, in GUI, "Payload contains"):
   - `transfer_frame` (default): Transfer Frame grezze, non codificate. RS,
     pseudo-randomizer e ASM vengono tutti applicati qui per costruire i CADU.
   - `cadu`: CADU gia' completi (ASM + blocco RS-codificato, gia'
     pseudo-randomizzato se e' cosi' che sono stati costruiti) -- ad es. CADU
     catturati o generati in precedenza. In questo caso RS, randomizer e ASM
     **non** vengono riapplicati (per evitare una doppia codifica e un secondo
     ASM davanti a dati che ne hanno gia' uno); viene comunque applicata,
     se abilitata, la sola codifica convoluzionale sul flusso di CADU, esattamente
     come farebbe un codificatore fisico a valle di un flusso di CADU gia' formato.

     Un file CADU reale/catturato non e' detto che inizi esattamente su un
     confine di CADU (idle non incorniciato davanti, o un estratto che parte a
     meta' flusso): con una sorgente reale, il tool cerca il primo ASM vero nel
     file prima di affettare (vedi `find_cadu_sync`), e valida che ogni CADU
     successivo inizi ancora con l'ASM atteso, segnalando un errore chiaro se
     la sincronizzazione si perde. La **lunghezza** di ogni CADU usata per
     l'affettamento viene inoltre *misurata dai dati stessi* (distanza fra i
     primi due ASM trovati nel file, vedi `detect_cadu_length`), non presa
     dalle impostazioni RS/interleave configurate in UI: un sistema reale non
     e' detto che usi esattamente la stessa codifica RS(255,*) interleaved di
     questo tool (campi extra, "virtual fill" CCSDS sezione 11, un altro
     sistema del tutto) -- fidarsi delle impostazioni configurate solo per la
     *lunghezza in byte* (non per la decodifica RS vera e propria, che in
     modalita' CADU viene comunque sempre saltata) disallineerebbe in modo
     silenzioso ogni CADU dopo il primo. Se la lunghezza misurata differisce
     da quella che le impostazioni RS-E/interleave predirebbero, viene
     mostrato un avviso (in GUI) o una riga di log (nel CLI) che lo segnala
     esplicitamente, e viene usata la lunghezza misurata.
2. **Reed-Solomon**: RS(255,223) con E=16 (default) o RS(255,239) con E=8,
   interleaving a profondita' I=1,2,3,4,5,8 selezionabile (byte `i` va nel
   sotto-stream `i mod I`). Implementazione CCSDS-nativa (non una libreria
   RS generica): campo di Galois GF(256) con polinomio F(x)=x^8+x^7+x^2+x+1
   (0x187, non lo 0x11D "generico"), polinomio generatore g(x) con radici
   alpha^(11j) preso direttamente dai coefficienti espansi in Annex G dello
   standard, e rappresentazione **dual-basis (Berlekamp)** obbligatoria
   (4.3.9) applicata alla sola parita' calcolata (i byte del Transfer Frame
   restano invariati, essendo la parte "non codificata" del CADU -- vedi
   Annex F, "Transformational Equivalence"). GF(256), coefficienti g(x) e
   trasformazione dual-basis sono stati verificati contro le tabelle di
   riferimento e gli esempi numerici forniti dallo standard stesso (Table
   F-1, Annex G, Examples 1-2 di Annex F), oltre che per divisibilita'
   algebrica del codeword risultante per tutte le 2E radici richieste.
3. **Pseudo-randomizer CCSDS** (opzionale, sezione 10): sequenza lunga a
   131071 bit (default dello standard dal 2023, polinomio x^17+x^14+1) o
   corta a 255 bit (legacy, polinomio 1+x^3+x^5+x^7+x^8) via XOR bit-a-bit
   al solo blocco RS-codificato di ogni CADU (mai all'ASM), reinizializzato
   a ogni CADU. Entrambe le sequenze sono state validate bit-per-bit contro
   le sequenze di riferimento (primi 40 bit) fornite dallo standard.
4. **ASM** (4 byte, `0x1ACFFC1D`) prepeso al blocco (randomizzato o meno) di
   ogni CADU -- questo e' letteralmente cio' che CCSDS chiama CADU.
5. **Convoluzionale** K=7, polinomi CCSDS 171/133 ottale, rate 1/2 (base) o
   punturato a 2/3, 3/4, 5/6, 7/8 (Table 3-1); l'inversione di G2 si applica
   solo a rate 1/2 (a rate punturati non c'e' inversione, per 3.4.1). Il
   codificatore lavora in modo continuo sull'intero flusso di CADU
   concatenati (ASM incluso), con stato del registro azzerato una sola
   volta all'inizio dell'intero segnale (non per-CADU). L'ASM viene quindi
   convoluzionalmente codificato insieme al resto: per un flusso
   convoluzionale la sincronizzazione di frame lato ricevitore si ottiene
   correlando l'ASM nel bitstream **gia' decodificato** (Viterbi decodifica
   in continuo, senza bisogno di sync preventiva), non prima della
   decodifica.
6. **NRZ-L**: mapping bipolare diretto (bit 1 -> +1, bit 0 -> -1), nessuna
   codifica differenziale.
7. **QPSK Gray**: coppie di campioni bipolari -> I/Q, normalizzati a
   energia media unitaria per simbolo.
8. **Pulse shaping RRC** (alpha configurabile, default 0.35).
9. **Normalizzazione**: il segnale finale viene scalato a un picco di
   ampiezza normalizzato configurabile (default 0.9 su scala [-1,+1]), per
   lasciare margine e prevenire saturazione/clipping in fase di playback RF.
10. **Resampling opzionale**: se richiesto (parametro `--target-fs` CLI o
    "Resample to fixed rate" in GUI), il segnale viene ricampionato
    (`scipy.signal.resample_poly`, rapporto intero esatto) a una frequenza
    di campionamento specifica accettata dallo strumento di playback,
    indipendente dalla frequenza nativa `symbol_rate x samples/symbol`
    usata internamente dalla catena.
11. **Output**: file raw IQ interleaved (`I0,Q0,I1,Q1,...`, nessun header),
    formato `float32` (default, range [-1,+1]) o `int16` (selezionabile;
    formato RF-Catcher/TestTree: little-endian, 12 bit significativi in
    complemento a 2 allineati LSB, range [-2048, 2047]), con file
    `.meta.json` affiancato contenente i parametri usati, la sample
    rate/formato effettivi del file esportato, e una nota che il file e'
    in banda base (nessuna informazione di frequenza portante: va
    impostata manualmente sullo strumento di playback, es. il campo TX
    Freq di RF-Catcher).

**Generazione a memoria costante (`export_chain`)**: sia il CLI sia il
pulsante "Generate export file" della GUI usano `ccsds_chain.pipeline.
export_chain()`, che processa le CADU a lotti (dimensionati per restare
intorno a ~64 MB di IQ nativo per lotto) invece di costruire in RAM gli
interi array bit/bit-codificati/simboli/IQ per l'intero export come fa
`run_chain()` (usata solo per l'anteprima live, volutamente limitata a un
numero ridotto di CADU). Il codificatore convoluzionale e il filtro RRC
mantengono lo stato tra un lotto e l'altro, cosi' il risultato e' identico
(byte-per-byte per `int16`; per `float32` puo' differire dall'ultimo bit
della mantissa per rumore di arrotondamento float64, ~9 ordini di
grandezza sotto il passo di quantizzazione int16) a una singola esecuzione
non a lotti. La memoria di picco resta quindi dell'ordine del singolo
lotto indipendentemente dalla durata dell'export -- un export che prima
esauriva la RAM (killed dall'OOM killer) ora la usa in modo trascurabile.
Il ricampionamento opzionale (`--target-fs`/"Resample to fixed rate") resta
l'unico stadio non a lotti (richiede l'intero segnale in memoria per
`scipy.signal.resample_poly`): un export troppo grande combinato con il
ricampionamento viene rifiutato subito con un messaggio chiaro, prima di
avviare la generazione, invece di fallire a meta' di un'esecuzione lunga.
In GUI il file esportato viene scritto su disco (cartella `output/`, come
il CLI) e offerto anche in download dal browser solo se resta sotto 1 GB
-- oltre quella soglia resta comunque disponibile al percorso mostrato in
pagina, perche' il pulsante di download di Streamlit deve comunque
caricare l'intero file in memoria per servirlo.

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
aggiorna in tempo reale**: spettro (PSD con banda -3dB/null-nullo evidenziata),
costellazione QPSK, estratto I/Q nel tempo, e le metriche (banda occupata,
durata segnale, tempo di calcolo). Include un indicatore visivo degli
stadi della pipeline attivi/disattivi ed export diretto del file IQ +
metadata dal browser (pulsanti "Scarica").

`app.py` e `generate_signal.py` condividono la stessa implementazione della
catena (`ccsds_chain/pipeline.py`): non c'e' rischio che GUI e CLI si
disallineino.

### CLI

```bash
python generate_signal.py                       # parametri baseline, salva in output/qpsk_ccsds_*.iq
python generate_signal.py --n-cadu 500 --randomizer long -o test.raw
python generate_signal.py --rs-e 8 --interleave-depth 2 --conv-rate 3/4 -o test2.raw
python generate_signal.py --dtype int16 --target-fs 10e6 --peak 0.9   # per un Recorder/Replayer RF

python verify_spectrum.py output/qpsk_ccsds_....iq --plot spectrum.png
```

I parametri baseline sono costanti in testa a `generate_signal.py`; le
opzioni piu' comuni sono anche esposte via CLI (`--help`), incluse quelle
di formato output (`--dtype`, `--peak`, `--target-fs`). Se `-o`/`--output`
non e' specificato, il file viene salvato in `output/` con nome
`qpsk_ccsds_<fs>Msps_<n_cadu>cadu_<timestamp>.iq`.

## Limitazioni

- **Turbo coding e LDPC** (sezioni 6, 7, 8 dello standard) non sono
  implementati: solo Reed-Solomon, convoluzionale (con puntura) e la loro
  concatenazione.
- **Transfer Frame lengths** (sezione 11): lo standard vincola le lunghezze
  di Transfer Frame ammesse per ciascuno schema di codifica (per garantire
  compatibilita' con la lunghezza del codeblock, tramite "virtual fill" se
  necessario). Questo non e' implementato: combinazioni di E / interleave
  depth / rate convoluzionale / N. CADU che producono un numero dispari di
  bit codificati falliscono (errore visibile in GUI/CLI, non un crash) sul
  pairing QPSK. Nella pratica capita solo con rate convoluzionali punturati
  in combinazioni particolari; la baseline (rate 1/2) non e' mai affetta.
- **Convenzione NRZ-L**: bit 1 -> +1, bit 0 -> -1; da verificare contro la
  polarita' attesa dal ricevitore/tool.
- **Formato file IQ**: raw interleaved (float32/int16, non-header) segue
  la specifica di formato IQ fornita per il Recorder/Replayer target;
  resta pero' da confermare con test end-to-end su hardware reale se lo
  specifico tool "IQ Converter" di RF-Catcher (TestTree) richiede un
  formato `.rfcatcher` diverso (con header/metadati propri) invece del
  raw binario prodotto qui.
- **Payload**: attualmente dati pseudo-casuali di test (o Transfer Frame
  grezzi da file); non viene costruito un vero header di Transfer Frame
  CCSDS (VCID, contatori, CRC, ecc.).
- **Transitorio filtro RRC**: essendo applicato un solo filtro RRC (non una
  coppia Tx/Rx accoppiata), l'uscita presenta un transitorio di
  `RRC_SPAN/2` simboli in testa e in coda.

## TODO

- [ ] Confermare formato input IQ Converter (contattare support@test-tree.com
      se necessario)
- [ ] Verificare polarita' NRZ-L attesa
- [ ] Implementare Turbo coding e LDPC (sezioni 6-8 CCSDS 131.0-B-5), se
      richiesti da un ricevitore/test specifico
- [ ] Implementare i vincoli di lunghezza Transfer Frame (sezione 11) e il
      "virtual fill" per RS (4.3.7-4.3.8), per evitare l'errore di parita'
      QPSK su combinazioni di parametri non standard
- [ ] Test end-to-end: generazione -> IQ Converter -> Capture & Playback ->
      loopback RX

## Verifica spettro

`verify_spectrum.py` calcola la PSD (media di periodogrammi con finestra di
Hanning) e misura la banda occupata a -3dB e la banda null-nullo (larghezza
del lobo principale tra i primi due null misurati sullo spettro) attorno al
centro banda. Con i parametri baseline (Rs=1.785 MS/s, alpha=0.35) ci si
attende una banda occupata a -3dB vicina a Rs (~1.7-1.8 MHz) e una banda
null-nullo vicina a Rs*(1+alpha) (~2.3-2.4 MHz, valore teorico esatto per un
filtro RRC ideale).
