s.boot;

// ── State & wavetable buffers ─────────────────────────────────────────────────
(
~numBufs = 32;
~bufSize = 16384;
~safetyMargin = 4; // slots either side of read head to avoid

b = Buffer.allocConsecutive(~numBufs, s, ~bufSize);
b.do(_.zero);

~readPos  = 0.0;
~writeIdx = 0;
~sampleBufs = []; // populated by the load block below

OSCdef(\readPos, { |msg|
    ~readPos = msg[3];
}, '/vosc_pos');

OSCdef(\recvWT, { |msg|
    var levels, times, curves, wt, candidate;

    levels = msg[1..];
    levels = [0] ++ levels ++ [0];
    levels = levels.normalize(-1, 1);
    times  = Array.fill(levels.size - 1, 1);
    curves = Array.fill(levels.size - 1, \lin);

    wt = Env.new(
        levels: levels,
        times:  times,
        curve:  curves
    ).asSignal(8192).normalize.asWavetable;

    // Advance writeIdx, skipping any slot too close to the read head
    candidate = (~writeIdx + 1) % ~numBufs;
    while {
        (candidate - ~readPos).abs % ~numBufs < ~safetyMargin
    } {
        candidate = (candidate + 1) % ~numBufs;
    };
    ~writeIdx = candidate;

    b[~writeIdx].loadCollection(wt);
    "Written to slot % (read head at %)".format(~writeIdx, ~readPos.round(0.01)).postln;
}, '/wavetable');

// Sample trigger: msg[1] = lateral position 0–1 (left→right within frame)
// Fired by Python when a passing signal crosses the global depth gradient threshold.
// Lateral position is derived from the local gradient-maxima column in the frame.
OSCdef(\sampleTrigger, { |msg|
    var pan, buf;
    if (~sampleBufs.size > 0) {
        pan = msg[1].linlin(0.0, 1.0, -1.0, 1.0);
        buf = ~sampleBufs.choose;
        Synth(\stationSample, [\buf, buf.bufnum, \pan, pan]);
        "Sample triggered — pan: %, buf: %".format(pan.round(0.01), buf.bufnum).postln;
    } {
        "Sample trigger received but no buffers loaded (run the load block).".postln;
    };
}, '/sample_trigger');
)

// ── Load station samples ──────────────────────────────────────────────────────
// Place mono .wav / .aif files in back-end/samples/ then evaluate this block.
(
var dir = thisProcess.nowExecutingPath.dirname +/+ "samples";
~sampleBufs = PathName(dir).files
    .select { |f| #["wav", "aif", "aiff"].includes(f.extension.toLower) }
    .collect { |f| Buffer.read(s, f.fullPath) };
"Loaded % sample(s) from %".format(~sampleBufs.size, dir).postln;
)

// ── SynthDef: Blackfriars station sample playback ─────────────────────────────
(
SynthDef(\stationSample, { |buf = 0, pan = 0, amp = 0.55, rel = 2.5|
    var sig, env;

    // Envelope: sustain at full amplitude for the sample duration, then fade
    // over `rel` seconds so the reverb tail rings out naturally.
    env = EnvGen.kr(
        Env([1, 1, 0], [BufDur.kr(buf), rel]),
        doneAction: 2
    );

    sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf));

    // Reverb for the release: extends spectral decay of each triggered sample
    sig = FreeVerb.ar(sig, mix: 0.45, room: 0.75, damp: 0.5);

    sig = Pan2.ar(sig, pan, amp * env);
    Out.ar(0, sig);
}).add;
)

// ── Main synthesis: VOsc + EQ + reverb ───────────────────────────────────────
(
x = {
    var sig, bufmod;

    bufmod = LFNoise1.kr(1/8 ! 8).range(0, ~numBufs - 1.001);

    // Report mean read position back at 20 Hz
    SendReply.kr(Impulse.kr(20), '/vosc_pos', bufmod.mean);

    sig = VOsc.ar(
        b[0].bufnum + bufmod,
        200 * { ExpRand(0.99, 1.01) }.dup(8),
        { Rand(0, 2pi) }.dup(8)
    );
    sig = Splay.ar(sig);
    sig = LeakDC.ar(sig);

    // EQ: parametric boost across C3–C6 (130.81–1046.5 Hz).
    // Centre = geometric mean ≈ 370 Hz; rq 2.0 gives a broad ~3-octave shelf;
    // +5 dB lift sets the melodic character described in the paper.
    sig = BPeakEQ.ar(sig, 370, 2.0, 5);

    // Reverb: sustains the spectral decay, establishing the ambient bed.
    sig = FreeVerb2.ar(sig[0], sig[1], mix: 0.35, room: 0.8, damp: 0.4);

    sig * 0.15;
}.play;

s.scope(zoom: 0.355);
)
