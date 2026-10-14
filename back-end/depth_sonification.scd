Server.default.record;

// ── State & wavetable buffers ─────────────────────────────────────────────────
(
// Boots the server if needed and waits until it's ready before allocating anything.
s.waitForBoot {
    ~numBufs = 32;
    ~bufSize = 16384;
    ~safetyMargin = 4; // slots either side of read head to avoid

    b = Buffer.allocConsecutive(~numBufs, s, ~bufSize);
    b.do(_.zero);

    ~readPos  = 0.0;
    ~writeIdx = 0;
    ~sampleBufs = []; // populatedx by the load block below

    // Band sliders: one control bus channel per vertical band of the frame.
    // Initialised to 0.5 so each mapping below sits at its original value until data arrives.
    ~numBands = 8;
    ~sliders = Bus.control(s, ~numBands);
    ~sliders.setn(0.5 ! ~numBands);

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

        b[~writeIdx].sendCollection(wt);
        "Written to slot % (read head at %)".format(~writeIdx, ~readPos.round(0.01)).postln;
    }, '/wavetable');

    // Band sliders: msg[1..] = mean contour height per band, 0 (bottom) – 1 (top), left→right
    OSCdef(\bands, { |msg|
        ~sliders.setn(msg[1..]);
    }, '/bands');

    // Slider smoothing: msg[1] = lag time in seconds (needs the main synth x running)
    OSCdef(\bandsLag, { |msg|
        x.set(\slLag, msg[1]);
    }, '/bands_lag');

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
};
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
    var sig, bufmod, sl, clicks;

    // Band sliders, lagged to glide between frames. 0.5 = original patch value.
    // Smoothing time in seconds is live-adjustable: x.set(\slLag, 2)
    sl = In.kr(~sliders.index, ~numBands).lag(\slLag.kr(0.5));

    // Band 6 → wavetable scan rate (1/256 – 4 Hz: near-frozen → rapid morphing)
    bufmod = LFNoise1.kr(sl[6].linexp(0, 1, 1/256, 4) ! 8).range(0, ~numBufs - 1.001);

    // Report mean read position back at 20 Hz
    SendReply.kr(Impulse.kr(20), '/vosc_pos', bufmod.mean);

    sig = VOsc.ar(
        b[0].bufnum + bufmod,
        // Band 7 → voice detune spread (±0.25% – ±4%: near-unison → wide chorus)
        200 * (1 + (sl[7].linexp(0, 1, 0.0025, 0.04) * { Rand(-1, 1) }.dup(8))),
        { Rand(0, 2pi) }.dup(8)
    );
    sig = Splay.ar(sig);
    sig = LeakDC.ar(sig);

    // Tight peak boost at C3–C6 (Q ≈ 20). Bands 0–3 scale each peak's gain 0–2×,
    // so the frame's left→right reads as the EQ's low→high.
    sig = BPeakEQ.ar(sig,  130.81, 0.05, sl[0].linlin(0, 1, 0, 24));
    sig = BPeakEQ.ar(sig,  261.63, 0.05, sl[1].linlin(0, 1, 0, 18));
    sig = BPeakEQ.ar(sig,  523.25, 0.05, sl[2].linlin(0, 1, 0, 12));
    sig = BPeakEQ.ar(sig, 1046.50, 0.05, sl[3].linlin(0, 1, 0,  6));

    // Clicks: one voice per band, panned to the band's position in the frame.
    // Click rate follows how far the band has moved from rest (0.5), so an empty
    // scene is silent. Pitch rises left→right. Level: x.set(\clickAmp, 0) to mute.
    clicks = ~numBands.collect { |i|
        var activity = ((sl[i] - 0.5).abs * 2).clip(0, 1);
        var trig     = Dust.ar(activity.squared * 32);   // up to ~16 clicks/s
        var click    = BPF.ar(Decay2.ar(trig, 0.0001, 0.004), i.linexp(0, ~numBands - 1, 600, 6000), 0.8);
        Pan2.ar(click, i.linlin(0, ~numBands - 1, -0.9, 0.9));
    }.sum;
    sig = sig + (clicks * \clickAmp.kr(4));

    // Reverb: sustains the spectral decay, establishing the ambient bed.
    // Band 4 → mix (0.1 – 0.6), band 5 → room size (0.6 – 1.0)
    sig = FreeVerb2.ar(sig[0], sig[1],
        mix:  1.0,
        room: sl[1].linlin(0, 1, 0.05, 1.0),
        damp: 0.1
    );

    sig * 0.25;
}.play;

s.scope(zoom: 0.355);
)

Server.default.stopRecording
