# %%
prompt_list= [
    {
        "clean_prompt": "Is 2824 > 1409? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1409 > 2824? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5012 > 4657? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4657 > 5012? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7912 > 1520? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1520 > 7912? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9279 > 1434? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1434 > 9279? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9928 > 7873? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7873 > 9928? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8359 > 5557? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5557 > 8359? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5552 > 3547? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3547 > 5552? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6514 > 2674? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2674 > 6514? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6635 > 5333? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5333 > 6635? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7201 > 2291? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2291 > 7201? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6925 > 4150? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4150 > 6925? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7227 > 5554? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5554 > 7227? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6977 > 3664? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3664 > 6977? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6820 > 4432? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4432 > 6820? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5422 > 4598? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4598 > 5422? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8517 > 3340? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3340 > 8517? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8019 > 7543? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7543 > 8019? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4593 > 3266? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3266 > 4593? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2489 > 1771? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1771 > 2489? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6573 > 2827? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2827 > 6573? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8123 > 3591? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3591 > 8123? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9317 > 2743? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2743 > 9317? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9317 > 4258? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4258 > 9317? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7126 > 3646? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3646 > 7126? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4923 > 1949? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1949 > 4923? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4770 > 4608? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4608 > 4770? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2163 > 1964? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1964 > 2163? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2104 > 1514? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1514 > 2104? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9834 > 3167? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3167 > 9834? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4119 > 2545? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2545 > 4119? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8062 > 6804? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6804 > 8062? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2612 > 1993? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1993 > 2612? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6559 > 2790? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2790 > 6559? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4139 > 4116? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4116 > 4139? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7396 > 5345? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5345 > 7396? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4566 > 1958? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1958 > 4566? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6138 > 1936? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1936 > 6138? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4044 > 2122? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2122 > 4044? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5033 > 1651? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1651 > 5033? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5272 > 4346? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4346 > 5272? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6180 > 2188? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2188 > 6180? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8508 > 2638? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2638 > 8508? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9808 > 4492? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4492 > 9808? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9666 > 1128? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1128 > 9666? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2891 > 2753? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2753 > 2891? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6617 > 4335? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4335 > 6617? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9280 > 9004? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 9004 > 9280? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5533 > 1722? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1722 > 5533? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6464 > 3143? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3143 > 6464? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7049 > 3426? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3426 > 7049? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 3088 > 1685? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1685 > 3088? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6974 > 1653? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1653 > 6974? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4878 > 3662? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3662 > 4878? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7755 > 1406? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1406 > 7755? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5371 > 3608? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3608 > 5371? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7267 > 1634? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1634 > 7267? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4644 > 4269? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4269 > 4644? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6728 > 6000? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6000 > 6728? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4652 > 1387? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1387 > 4652? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7528 > 6378? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6378 > 7528? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9346 > 7548? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7548 > 9346? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7311 > 4114? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4114 > 7311? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6409 > 6143? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6143 > 6409? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7691 > 6344? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6344 > 7691? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5844 > 3085? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3085 > 5844? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7888 > 7211? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7211 > 7888? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5978 > 5700? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5700 > 5978? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8043 > 6279? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6279 > 8043? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9375 > 8752? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8752 > 9375? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4680 > 4262? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4262 > 4680? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8784 > 2193? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2193 > 8784? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7790 > 4185? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4185 > 7790? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9099 > 7547? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7547 > 9099? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 3417 > 1090? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1090 > 3417? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7965 > 4585? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4585 > 7965? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9486 > 8611? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8611 > 9486? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5082 > 2988? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2988 > 5082? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9976 > 8305? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8305 > 9976? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8777 > 8373? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8373 > 8777? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9849 > 2320? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2320 > 9849? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7797 > 7678? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 7678 > 7797? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9890 > 8633? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8633 > 9890? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7381 > 1320? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1320 > 7381? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8814 > 1096? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1096 > 8814? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8999 > 4595? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4595 > 8999? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7371 > 6507? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6507 > 7371? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9751 > 1441? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1441 > 9751? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 6363 > 4467? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4467 > 6363? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9837 > 1853? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1853 > 9837? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4673 > 2124? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2124 > 4673? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7043 > 3749? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3749 > 7043? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7498 > 4249? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4249 > 7498? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 4978 > 2669? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2669 > 4978? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 2983 > 1672? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1672 > 2983? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9728 > 8018? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8018 > 9728? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8532 > 3506? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 3506 > 8532? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9817 > 8921? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8921 > 9817? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8136 > 5397? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5397 > 8136? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5022 > 2419? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 2419 > 5022? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8385 > 4995? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4995 > 8385? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7209 > 6511? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6511 > 7209? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9098 > 6325? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6325 > 9098? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8988 > 4475? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4475 > 8988? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5526 > 1166? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 1166 > 5526? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9004 > 4937? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4937 > 9004? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9041 > 8342? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 8342 > 9041? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7625 > 4986? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4986 > 7625? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 7971 > 6419? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 6419 > 7971? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 8434 > 5438? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5438 > 8434? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 5118 > 4777? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4777 > 5118? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9779 > 4033? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 4033 > 9779? Answer:",
        "clean_label": " No",
    },
    {
        "clean_prompt": "Is 9595 > 5636? Answer:",
        "clean_label": " Yes",
    },
    {
        "clean_prompt": "Is 5636 > 9595? Answer:",
        "clean_label": " No",
    },
]
