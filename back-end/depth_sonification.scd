s.boot;

(
~numBufs = 32;
~bufSize = 16384;
~safetyMargin = 4; // slots either side of read head to avoid

b = Buffer.allocConsecutive(~numBufs, s, ~bufSize);
b.do(_.zero);

~readPos = 0.0;
~writeIdx = 0;

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
)

(
x = {
    var sig, bufmod;

    bufmod = LFNoise1.kr(1/8 ! 8).range(0, ~numBufs - 1.001);

    // Report mean read position back at 20hz
    SendReply.kr(Impulse.kr(20), '/vosc_pos', bufmod.mean);

    sig = VOsc.ar(
        b[0].bufnum + bufmod,
        200 * { ExpRand(0.99, 1.01) }.dup(8),
        { Rand(0, 2pi) }.dup(8)
    );
    sig = Splay.ar(sig);
    sig = LeakDC.ar(sig);
    sig * 0.15;
}.play;

s.scope(zoom: 0.355);
)