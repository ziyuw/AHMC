/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/net/net-spec ./neal.net 38 20 8 1 / - 0.1:1:2 0.2:1 - x0.3:1 - 0.2:1 - x0.1:1:4 - - 10
/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/util/model-spec ./neal.net binary
/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/util/data-spec ./neal.net 38 1 2 / train.data.sel train.classes valid.data.sel valid.classes / -500x0.05 ...

/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/net/net-gen ./neal.net fix 0.5

/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/mc/mc-spec ./neal.net repeat 40 heatbath hybrid 100:4 0.02
/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/net/net-mc ./neal.net 1

/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/mc/mc-spec ./neal.net repeat 40 sample-sigmas heatbath hybrid 100:4 0.03
/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/net/net-mc ./neal.net 2

/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/mc/mc-spec ./neal.net repeat 5 sample-sigmas heatbath 0.8 hybrid 800:8 0.05 negate
/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/net/net-mc ./neal.net 200