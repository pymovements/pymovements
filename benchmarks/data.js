window.BENCHMARK_DATA = {
  "lastUpdate": 1795625711853,
  "repoUrl": "https://github.com/pymovements/pymovements",
  "entries": {
    "pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e6304f2cd784bf90d1d47a13b5705b2457210484",
          "message": "ci: publish benchmark history as a chart on GitHub Pages (#1807)",
          "timestamp": "2026-11-12T16:19:15+01:00",
          "tree_id": "cefb2eaabca25a08598e6c03f514e76f51ffa353",
          "url": "https://github.com/pymovements/pymovements/commit/e6304f2cd784bf90d1d47a13b5705b2457210484"
        },
        "date": 1794497501289,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 28.792854258530703,
            "unit": "iter/sec",
            "range": "stddev: 0.006736321263423959",
            "extra": "mean: 34.73083950000273 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.2304168505079875,
            "unit": "iter/sec",
            "range": "stddev: 0.02060803536979308",
            "extra": "mean: 138.30461240000886 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 13.845308070364158,
            "unit": "iter/sec",
            "range": "stddev: 0.00862751049193101",
            "extra": "mean: 72.22663410000223 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.166983470685282,
            "unit": "iter/sec",
            "range": "stddev: 0.03820739786153168",
            "extra": "mean: 315.75788420001345 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 7.1360006287911055,
            "unit": "iter/sec",
            "range": "stddev: 0.031437980930565386",
            "extra": "mean: 140.1345167999807 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 7.732674190700837,
            "unit": "iter/sec",
            "range": "stddev: 0.017150106560641008",
            "extra": "mean: 129.3213673999844 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.4664954483881831,
            "unit": "iter/sec",
            "range": "stddev: 0.015449921191675481",
            "extra": "mean: 2.143643637800028 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.2804573785085233,
            "unit": "iter/sec",
            "range": "stddev: 0.03138565629453426",
            "extra": "mean: 3.565604175999988 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.11216480514478529,
            "unit": "iter/sec",
            "range": "stddev: 0.034784600061146205",
            "extra": "mean: 8.91545256740003 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.15417661988423206,
            "unit": "iter/sec",
            "range": "stddev: 0.019204847310178113",
            "extra": "mean: 6.486067736800033 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.3489482554505188,
            "unit": "iter/sec",
            "range": "stddev: 0.053735936731135005",
            "extra": "mean: 741.3182796000001 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.3665432428678577,
            "unit": "iter/sec",
            "range": "stddev: 0.05778156808026035",
            "extra": "mean: 731.7734036000047 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.3150735175530082,
            "unit": "iter/sec",
            "range": "stddev: 0.039853337184417414",
            "extra": "mean: 760.413761399991 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 26.130804420525976,
            "unit": "iter/sec",
            "range": "stddev: 0.004588904498266096",
            "extra": "mean: 38.26900940005089 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 38.17943997754056,
            "unit": "iter/sec",
            "range": "stddev: 0.006320252823130045",
            "extra": "mean: 26.192107600013514 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 37.26319731839115,
            "unit": "iter/sec",
            "range": "stddev: 0.004945418861048472",
            "extra": "mean: 26.83612980001726 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 35.63820095540287,
            "unit": "iter/sec",
            "range": "stddev: 0.006298959389592569",
            "extra": "mean: 28.059777799990115 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.47869254935299127,
            "unit": "iter/sec",
            "range": "stddev: 0.009476926017589518",
            "extra": "mean: 2.0890235316000143 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 0.9710700788906272,
            "unit": "iter/sec",
            "range": "stddev: 0.020621877992604065",
            "extra": "mean: 1.0297917954000013 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.4840839385656111,
            "unit": "iter/sec",
            "range": "stddev: 0.04171033464910476",
            "extra": "mean: 2.065757444799965 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 0.9829157139395903,
            "unit": "iter/sec",
            "range": "stddev: 0.02172810377731122",
            "extra": "mean: 1.0173812319999798 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.4692079468681634,
            "unit": "iter/sec",
            "range": "stddev: 0.1396528325834937",
            "extra": "mean: 2.1312512004000155 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 0.9727232856317616,
            "unit": "iter/sec",
            "range": "stddev: 0.01866315816517945",
            "extra": "mean: 1.0280415970000376 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 26.135655342878163,
            "unit": "iter/sec",
            "range": "stddev: 0.0010644052136069126",
            "extra": "mean: 38.261906459999864 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.6210834878378926,
            "unit": "iter/sec",
            "range": "stddev: 0.004630702356566502",
            "extra": "mean: 1.6100894961500045 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.17342836800745,
            "unit": "iter/sec",
            "range": "stddev: 0.007509973387087232",
            "extra": "mean: 460.1025801999526 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 144.35512006269337,
            "unit": "iter/sec",
            "range": "stddev: 0.0019245828588174236",
            "extra": "mean: 6.927360799988946 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 108.78592912304876,
            "unit": "iter/sec",
            "range": "stddev: 0.0026214579422660267",
            "extra": "mean: 9.192365300009442 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71385469+kirthan98@users.noreply.github.com",
            "name": "kirthan98",
            "username": "kirthan98"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "51440f4980a98d455e9dfe739d42cae1bff16ce0",
          "message": "fix: require the dataclass decorator on synthetic generator subclasses (#1821)\n\nCo-authored-by: Daniel G. Krakowczyk <daniel.krakowczyk@uni-potsdam.de>",
          "timestamp": "2026-11-17T17:59:37Z",
          "tree_id": "c4db33d1d519d3440010b83b692b30cb60e673c1",
          "url": "https://github.com/pymovements/pymovements/commit/51440f4980a98d455e9dfe739d42cae1bff16ce0"
        },
        "date": 1794939065799,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 28.956965414721807,
            "unit": "iter/sec",
            "range": "stddev: 0.005804499993652826",
            "extra": "mean: 34.53400540001326 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.002440136813731,
            "unit": "iter/sec",
            "range": "stddev: 0.018600740079404748",
            "extra": "mean: 142.8073614999903 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 14.003744195089121,
            "unit": "iter/sec",
            "range": "stddev: 0.007189247467928578",
            "extra": "mean: 71.40947350000033 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.137863334399698,
            "unit": "iter/sec",
            "range": "stddev: 0.03902429692829722",
            "extra": "mean: 318.6881942999946 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 8.667446343912351,
            "unit": "iter/sec",
            "range": "stddev: 0.041661392218089024",
            "extra": "mean: 115.37423600001375 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 8.584514304679878,
            "unit": "iter/sec",
            "range": "stddev: 0.004772363079420907",
            "extra": "mean: 116.4888268000027 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.48092188800068286,
            "unit": "iter/sec",
            "range": "stddev: 0.034901813182675386",
            "extra": "mean: 2.079339753399995 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.28387301073240495,
            "unit": "iter/sec",
            "range": "stddev: 0.020442229021085563",
            "extra": "mean: 3.5227019202000065 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.1222517091801673,
            "unit": "iter/sec",
            "range": "stddev: 0.04934757932115237",
            "extra": "mean: 8.179844737600025 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.16396745202684973,
            "unit": "iter/sec",
            "range": "stddev: 0.025708840157141474",
            "extra": "mean: 6.098771357599981 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.2717128076743311,
            "unit": "iter/sec",
            "range": "stddev: 0.05045575880666805",
            "extra": "mean: 786.3410622000174 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.3187795641068585,
            "unit": "iter/sec",
            "range": "stddev: 0.04480167209700242",
            "extra": "mean: 758.2768395999892 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.1982694017172832,
            "unit": "iter/sec",
            "range": "stddev: 0.06089169134109978",
            "extra": "mean: 834.5368733999749 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 30.715440377872085,
            "unit": "iter/sec",
            "range": "stddev: 0.0051061220008362055",
            "extra": "mean: 32.556915600025604 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 38.70283575679431,
            "unit": "iter/sec",
            "range": "stddev: 0.00535280457571752",
            "extra": "mean: 25.837899999987712 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 38.452233908333575,
            "unit": "iter/sec",
            "range": "stddev: 0.003688738980830172",
            "extra": "mean: 26.006291399971815 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 28.653099072784496,
            "unit": "iter/sec",
            "range": "stddev: 0.006846092138196767",
            "extra": "mean: 34.90023879999171 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.48872944162536097,
            "unit": "iter/sec",
            "range": "stddev: 0.011230093671881344",
            "extra": "mean: 2.046121872000003 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 0.9537519508268971,
            "unit": "iter/sec",
            "range": "stddev: 0.025977421880976664",
            "extra": "mean: 1.0484906469999942 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.48759665453557316,
            "unit": "iter/sec",
            "range": "stddev: 0.032337994786631644",
            "extra": "mean: 2.05087543300001 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 0.955159374550187,
            "unit": "iter/sec",
            "range": "stddev: 0.027039818054309536",
            "extra": "mean: 1.046945699999992 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.4791000419847592,
            "unit": "iter/sec",
            "range": "stddev: 0.04021854812687254",
            "extra": "mean: 2.0872467384000175 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 0.9756389239891865,
            "unit": "iter/sec",
            "range": "stddev: 0.003941066547187675",
            "extra": "mean: 1.0249693564000153 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 29.538814203910913,
            "unit": "iter/sec",
            "range": "stddev: 0.0007676970214704882",
            "extra": "mean: 33.853762479998295 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.7245558641799785,
            "unit": "iter/sec",
            "range": "stddev: 0.003461199969947703",
            "extra": "mean: 1.380155829849997 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.2529845300715334,
            "unit": "iter/sec",
            "range": "stddev: 0.010517120875983908",
            "extra": "mean: 443.85568859997875 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 140.86967896659775,
            "unit": "iter/sec",
            "range": "stddev: 0.0017368012736061105",
            "extra": "mean: 7.098759699999846 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 113.48159150968519,
            "unit": "iter/sec",
            "range": "stddev: 0.0027286036001818214",
            "extra": "mean: 8.812001900014366 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "66853113+pre-commit-ci[bot]@users.noreply.github.com",
            "name": "pre-commit-ci[bot]",
            "username": "pre-commit-ci[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b0cdb85a7b75c82154360fcb8c9011d3aa3baedd",
          "message": "ci: pre-commit autoupdate (#1838)\n\nupdates:\n- [github.com/pappasam/toml-sort: v0.25.0 → v0.26.0](https://github.com/pappasam/toml-sort/compare/v0.25.0...v0.26.0)\n- [github.com/PyCQA/pylint: v4.1.3 → v4.1.4](https://github.com/PyCQA/pylint/compare/v4.1.3...v4.1.4)",
          "timestamp": "2026-11-17T20:50:09+01:00",
          "tree_id": "3dc6c191e9d5c9b5c9ee19ea89a32774e01ed237",
          "url": "https://github.com/pymovements/pymovements/commit/b0cdb85a7b75c82154360fcb8c9011d3aa3baedd"
        },
        "date": 1794945432662,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 45.98420384658296,
            "unit": "iter/sec",
            "range": "stddev: 0.004697075133172228",
            "extra": "mean: 21.746598099997527 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 11.483963031487116,
            "unit": "iter/sec",
            "range": "stddev: 0.013963945195883388",
            "extra": "mean: 87.07795360000432 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 19.92066305097933,
            "unit": "iter/sec",
            "range": "stddev: 0.00548660885027104",
            "extra": "mean: 50.19913230000839 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 4.54520947395772,
            "unit": "iter/sec",
            "range": "stddev: 0.023865166102537586",
            "extra": "mean: 220.01186210000014 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 10.723056676788698,
            "unit": "iter/sec",
            "range": "stddev: 0.02416030258928183",
            "extra": "mean: 93.25699100002112 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 10.207553950862998,
            "unit": "iter/sec",
            "range": "stddev: 0.016686696719195974",
            "extra": "mean: 97.96666319999758 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.8094771382868036,
            "unit": "iter/sec",
            "range": "stddev: 0.018639139623604747",
            "extra": "mean: 1.2353653397999893 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.4706227126489586,
            "unit": "iter/sec",
            "range": "stddev: 0.06850805810902234",
            "extra": "mean: 2.1248443246000077 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.2204689694245174,
            "unit": "iter/sec",
            "range": "stddev: 0.08382318841388975",
            "extra": "mean: 4.535785705400019 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.29976433775983874,
            "unit": "iter/sec",
            "range": "stddev: 0.05785676429453835",
            "extra": "mean: 3.3359538612000166 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.1098270113168804,
            "unit": "iter/sec",
            "range": "stddev: 0.040422495954174895",
            "extra": "mean: 473.9725079999971 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 2.1793871513728154,
            "unit": "iter/sec",
            "range": "stddev: 0.03632192664668396",
            "extra": "mean: 458.844588199986 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.97797787867915,
            "unit": "iter/sec",
            "range": "stddev: 0.03809999425880116",
            "extra": "mean: 505.56682699999556 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 49.82094849318493,
            "unit": "iter/sec",
            "range": "stddev: 0.0025776290432013033",
            "extra": "mean: 20.0718780000102 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 56.02558657158586,
            "unit": "iter/sec",
            "range": "stddev: 0.00399459188744595",
            "extra": "mean: 17.848987600018518 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 50.234447683761985,
            "unit": "iter/sec",
            "range": "stddev: 0.0035113942488476193",
            "extra": "mean: 19.906658599995808 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 60.881599656727545,
            "unit": "iter/sec",
            "range": "stddev: 0.002571968698081983",
            "extra": "mean: 16.425323999999364 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.7762807020374285,
            "unit": "iter/sec",
            "range": "stddev: 0.0217538439384056",
            "extra": "mean: 1.2881938161999869 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 1.531061644767878,
            "unit": "iter/sec",
            "range": "stddev: 0.013548942896979143",
            "extra": "mean: 653.1415657999901 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.7788326041149012,
            "unit": "iter/sec",
            "range": "stddev: 0.010836435365830947",
            "extra": "mean: 1.283972954800015 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 1.5379083868853665,
            "unit": "iter/sec",
            "range": "stddev: 0.006720021473337229",
            "extra": "mean: 650.233790599998 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.7685512573560691,
            "unit": "iter/sec",
            "range": "stddev: 0.0842456714751596",
            "extra": "mean: 1.3011493903999962 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 1.4772354965401224,
            "unit": "iter/sec",
            "range": "stddev: 0.026613967503889866",
            "extra": "mean: 676.9401373999813 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 47.913637500625306,
            "unit": "iter/sec",
            "range": "stddev: 0.0014998733809885542",
            "extra": "mean: 20.870884619998833 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 1.233424868060235,
            "unit": "iter/sec",
            "range": "stddev: 0.015379366620961716",
            "extra": "mean: 810.7506390500021 msec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 3.185971099780831,
            "unit": "iter/sec",
            "range": "stddev: 0.012244662080908193",
            "extra": "mean: 313.8760424000054 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 252.8919522418068,
            "unit": "iter/sec",
            "range": "stddev: 0.0013397027935988372",
            "extra": "mean: 3.9542579000055866 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 182.13691811335462,
            "unit": "iter/sec",
            "range": "stddev: 0.001887271096888655",
            "extra": "mean: 5.490375099998346 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94641613+RClaudiuM@users.noreply.github.com",
            "name": "Roman Claudiu (ClaW)",
            "username": "RClaudiuM"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8061736b6c9e4c4d475fb199836408d93122978a",
          "message": "fix: apply screen pixel axes limits only to pixel columns (#1823)",
          "timestamp": "2026-11-17T21:24:32+01:00",
          "tree_id": "998f3311325ce72ba19dfa46feb2579624dd93b6",
          "url": "https://github.com/pymovements/pymovements/commit/8061736b6c9e4c4d475fb199836408d93122978a"
        },
        "date": 1794947473828,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 49.14069992233619,
            "unit": "iter/sec",
            "range": "stddev: 0.004364527406265917",
            "extra": "mean: 20.349730499981433 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 12.284334865818666,
            "unit": "iter/sec",
            "range": "stddev: 0.013806360705121114",
            "extra": "mean: 81.40448880000122 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 20.935134743901045,
            "unit": "iter/sec",
            "range": "stddev: 0.004991658320114064",
            "extra": "mean: 47.766590099990935 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 4.508276687238976,
            "unit": "iter/sec",
            "range": "stddev: 0.04043950685470097",
            "extra": "mean: 221.81424729998866 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 11.898899996670961,
            "unit": "iter/sec",
            "range": "stddev: 0.02673488822030978",
            "extra": "mean: 84.04138199999807 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 12.498531734985546,
            "unit": "iter/sec",
            "range": "stddev: 0.012929183582942136",
            "extra": "mean: 80.00939799999287 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.8206908401571671,
            "unit": "iter/sec",
            "range": "stddev: 0.023997067621629595",
            "extra": "mean: 1.2184856355999956 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.4963004652106438,
            "unit": "iter/sec",
            "range": "stddev: 0.03366138834522293",
            "extra": "mean: 2.014908447800008 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.23397903515471755,
            "unit": "iter/sec",
            "range": "stddev: 0.05184794700031245",
            "extra": "mean: 4.273887185399985 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.32386885694191453,
            "unit": "iter/sec",
            "range": "stddev: 0.060326570847251224",
            "extra": "mean: 3.0876695259999907 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.1166185830966535,
            "unit": "iter/sec",
            "range": "stddev: 0.02397002474519477",
            "extra": "mean: 472.45167739998806 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 2.1406762057307365,
            "unit": "iter/sec",
            "range": "stddev: 0.0341142997506955",
            "extra": "mean: 467.14211020000675 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 2.1217489335937807,
            "unit": "iter/sec",
            "range": "stddev: 0.028453140697322646",
            "extra": "mean: 471.3092978000077 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 40.15086317452448,
            "unit": "iter/sec",
            "range": "stddev: 0.002254664993010041",
            "extra": "mean: 24.90606479998405 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 64.30313134000217,
            "unit": "iter/sec",
            "range": "stddev: 0.0033656876124591464",
            "extra": "mean: 15.551342200001272 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 61.48364536876288,
            "unit": "iter/sec",
            "range": "stddev: 0.0017106859510656362",
            "extra": "mean: 16.264487800003735 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 65.23486488838587,
            "unit": "iter/sec",
            "range": "stddev: 0.0034347233155523527",
            "extra": "mean: 15.329226199992263 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.8186166070286377,
            "unit": "iter/sec",
            "range": "stddev: 0.02837709528741657",
            "extra": "mean: 1.2215730678 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 1.593994773379071,
            "unit": "iter/sec",
            "range": "stddev: 0.019428250826875083",
            "extra": "mean: 627.3546292000219 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.8165874391648519,
            "unit": "iter/sec",
            "range": "stddev: 0.0382224626768802",
            "extra": "mean: 1.2246085992000189 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 1.5939307589904748,
            "unit": "iter/sec",
            "range": "stddev: 0.01640355673363725",
            "extra": "mean: 627.3798246000069 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.8187604775959569,
            "unit": "iter/sec",
            "range": "stddev: 0.011693804088578902",
            "extra": "mean: 1.2213584159999982 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 1.5327974843570678,
            "unit": "iter/sec",
            "range": "stddev: 0.030713608701963897",
            "extra": "mean: 652.4019058000022 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 51.53589282858695,
            "unit": "iter/sec",
            "range": "stddev: 0.0009976849480745414",
            "extra": "mean: 19.403952180009583 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 1.2857307494689671,
            "unit": "iter/sec",
            "range": "stddev: 0.010408674138968526",
            "extra": "mean: 777.7678183500086 msec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 3.2440069873483184,
            "unit": "iter/sec",
            "range": "stddev: 0.015988710546916387",
            "extra": "mean: 308.26074170000766 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 266.2565534738925,
            "unit": "iter/sec",
            "range": "stddev: 0.0013266692473627628",
            "extra": "mean: 3.7557760999789025 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 184.68579479565443,
            "unit": "iter/sec",
            "range": "stddev: 0.0018747503659659983",
            "extra": "mean: 5.4146016000117925 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49326130+cbueth@users.noreply.github.com",
            "name": "Carlson Büth",
            "username": "cbueth"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "67bf0d643733e680f7426ab1ed25068bcabfe248",
          "message": "feat: add word_column to fixation correction (#1835)",
          "timestamp": "2026-11-18T12:10:53Z",
          "tree_id": "9ef05cf52aab5370100cd8792e08f276604feb0a",
          "url": "https://github.com/pymovements/pymovements/commit/67bf0d643733e680f7426ab1ed25068bcabfe248"
        },
        "date": 1795004316552,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 37.663521025728045,
            "unit": "iter/sec",
            "range": "stddev: 0.00477926752384264",
            "extra": "mean: 26.550889899988306 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 9.080588902559546,
            "unit": "iter/sec",
            "range": "stddev: 0.013520368009984839",
            "extra": "mean: 110.1250162000099 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 17.007136153512704,
            "unit": "iter/sec",
            "range": "stddev: 0.0058453051730169205",
            "extra": "mean: 58.79884720000064 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.8579556738408325,
            "unit": "iter/sec",
            "range": "stddev: 0.023610917263891907",
            "extra": "mean: 259.20463699999914 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 8.37055337559335,
            "unit": "iter/sec",
            "range": "stddev: 0.027369642510881614",
            "extra": "mean: 119.46641459999228 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.3498401994501,
            "unit": "iter/sec",
            "range": "stddev: 0.010415042767177139",
            "extra": "mean: 106.95369960000107 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.8485022729161378,
            "unit": "iter/sec",
            "range": "stddev: 0.02008801349097616",
            "extra": "mean: 1.1785472259999892 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.48713672972231276,
            "unit": "iter/sec",
            "range": "stddev: 0.014716773664802158",
            "extra": "mean: 2.052811744600001 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.18232921457174364,
            "unit": "iter/sec",
            "range": "stddev: 0.062179071150651556",
            "extra": "mean: 5.484584586999995 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.25610156643596566,
            "unit": "iter/sec",
            "range": "stddev: 0.024908249246977824",
            "extra": "mean: 3.9047008337999953 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.8262209130334917,
            "unit": "iter/sec",
            "range": "stddev: 0.05095122410280138",
            "extra": "mean: 547.5788788000045 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.8210546736406596,
            "unit": "iter/sec",
            "range": "stddev: 0.02814778815642925",
            "extra": "mean: 549.1323321999971 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.7597394430109266,
            "unit": "iter/sec",
            "range": "stddev: 0.02151971206749928",
            "extra": "mean: 568.2659463999926 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 32.456422254840824,
            "unit": "iter/sec",
            "range": "stddev: 0.004273216827440393",
            "extra": "mean: 30.81054319999339 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 37.582511276006706,
            "unit": "iter/sec",
            "range": "stddev: 0.004103633479376214",
            "extra": "mean: 26.608120800017332 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 43.330320605027424,
            "unit": "iter/sec",
            "range": "stddev: 0.002514259863652173",
            "extra": "mean: 23.078527600000598 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 39.7405416338392,
            "unit": "iter/sec",
            "range": "stddev: 0.0022117622716361037",
            "extra": "mean: 25.163220200010983 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.7072621265936805,
            "unit": "iter/sec",
            "range": "stddev: 0.055302660557716835",
            "extra": "mean: 1.4139029397999934 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 1.3994459636533128,
            "unit": "iter/sec",
            "range": "stddev: 0.01023732669127106",
            "extra": "mean: 714.5684977999849 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.7059020015988154,
            "unit": "iter/sec",
            "range": "stddev: 0.05656427035343189",
            "extra": "mean: 1.4166272340000092 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 1.4109178056896836,
            "unit": "iter/sec",
            "range": "stddev: 0.004719001898633014",
            "extra": "mean: 708.7585088000083 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.7119407067973772,
            "unit": "iter/sec",
            "range": "stddev: 0.026298160270432618",
            "extra": "mean: 1.40461135380001 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 1.414517270354686,
            "unit": "iter/sec",
            "range": "stddev: 0.004768049253482779",
            "extra": "mean: 706.954959799998 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 49.79355839656251,
            "unit": "iter/sec",
            "range": "stddev: 0.0003030402397569033",
            "extra": "mean: 20.08291900000131 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 1.2279240569028256,
            "unit": "iter/sec",
            "range": "stddev: 0.0012641567128311745",
            "extra": "mean: 814.3826113499927 msec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 3.230910259057681,
            "unit": "iter/sec",
            "range": "stddev: 0.002882315504018071",
            "extra": "mean: 309.51029889999404 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 169.57459398136743,
            "unit": "iter/sec",
            "range": "stddev: 0.0017891694036873886",
            "extra": "mean: 5.897109800008593 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 140.17008237797103,
            "unit": "iter/sec",
            "range": "stddev: 0.002141827805241978",
            "extra": "mean: 7.1341899999993075 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1c5e2a0ffc3cccf96f9f6ff4a6c8414882bd2df9",
          "message": "feat!: add durations and offset convention to `Events` (#1806)",
          "timestamp": "2026-11-18T15:25:09+01:00",
          "tree_id": "92bd6ff041fa29f50ae1ed0ed76eef31eb4d3314",
          "url": "https://github.com/pymovements/pymovements/commit/1c5e2a0ffc3cccf96f9f6ff4a6c8414882bd2df9"
        },
        "date": 1795012469612,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 35.46429976510943,
            "unit": "iter/sec",
            "range": "stddev: 0.004781264856562469",
            "extra": "mean: 28.197370500004126 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 8.252124785354173,
            "unit": "iter/sec",
            "range": "stddev: 0.014643059970704683",
            "extra": "mean: 121.18091109998659 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 16.265904069031993,
            "unit": "iter/sec",
            "range": "stddev: 0.006663516666417412",
            "extra": "mean: 61.478292000003876 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.772558075143729,
            "unit": "iter/sec",
            "range": "stddev: 0.03311387447659487",
            "extra": "mean: 265.07212880000566 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 10.037405253895198,
            "unit": "iter/sec",
            "range": "stddev: 0.025989597253397532",
            "extra": "mean: 99.62734140000293 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.705866435319116,
            "unit": "iter/sec",
            "range": "stddev: 0.010977468771806342",
            "extra": "mean: 103.03047200001174 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.6176347369524018,
            "unit": "iter/sec",
            "range": "stddev: 0.024519886191562063",
            "extra": "mean: 1.6190799191999872 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.3632520829035924,
            "unit": "iter/sec",
            "range": "stddev: 0.006638158906058996",
            "extra": "mean: 2.752909197399981 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.138293741687026,
            "unit": "iter/sec",
            "range": "stddev: 0.015138520288660827",
            "extra": "mean: 7.230985204399997 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.16690091839223067,
            "unit": "iter/sec",
            "range": "stddev: 0.027899415109861093",
            "extra": "mean: 5.991578774000027 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.6609514340999807,
            "unit": "iter/sec",
            "range": "stddev: 0.06623101548420361",
            "extra": "mean: 602.0645633999948 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.6461534183110462,
            "unit": "iter/sec",
            "range": "stddev: 0.03836600396671776",
            "extra": "mean: 607.4767933999738 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.568144626752338,
            "unit": "iter/sec",
            "range": "stddev: 0.047476840235495185",
            "extra": "mean: 637.696283199989 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 37.133487557648536,
            "unit": "iter/sec",
            "range": "stddev: 0.0036697030352557027",
            "extra": "mean: 26.929870199978723 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 49.88449843002373,
            "unit": "iter/sec",
            "range": "stddev: 0.0030469965640353",
            "extra": "mean: 20.046307600000546 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 46.61854743714023,
            "unit": "iter/sec",
            "range": "stddev: 0.005354042986026504",
            "extra": "mean: 21.450689799985412 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 46.82005534828624,
            "unit": "iter/sec",
            "range": "stddev: 0.002872787669195223",
            "extra": "mean: 21.35836859997653 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.6128052894870006,
            "unit": "iter/sec",
            "range": "stddev: 0.01632217051088626",
            "extra": "mean: 1.6318397003999963 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 1.1933700275523311,
            "unit": "iter/sec",
            "range": "stddev: 0.016377543164227067",
            "extra": "mean: 837.96305999997 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.6233996170747962,
            "unit": "iter/sec",
            "range": "stddev: 0.024838394448468994",
            "extra": "mean: 1.6041074979999848 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 1.2081171892984581,
            "unit": "iter/sec",
            "range": "stddev: 0.006473181986144637",
            "extra": "mean: 827.7342701999714 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.6158300402882184,
            "unit": "iter/sec",
            "range": "stddev: 0.005750472569192788",
            "extra": "mean: 1.6238246505999996 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 1.1979607572130249,
            "unit": "iter/sec",
            "range": "stddev: 0.013866429447313536",
            "extra": "mean: 834.7518847999936 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 37.467410555931075,
            "unit": "iter/sec",
            "range": "stddev: 0.00013418457681178142",
            "extra": "mean: 26.6898615399964 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.9144047931567453,
            "unit": "iter/sec",
            "range": "stddev: 0.005356894230231424",
            "extra": "mean: 1.0936075657999993 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.909685813083019,
            "unit": "iter/sec",
            "range": "stddev: 0.0028113035321965254",
            "extra": "mean: 343.6797181000202 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 185.49603876006273,
            "unit": "iter/sec",
            "range": "stddev: 0.0014849660557850264",
            "extra": "mean: 5.390950699995756 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 143.8158418582552,
            "unit": "iter/sec",
            "range": "stddev: 0.0017734079526972916",
            "extra": "mean: 6.953336900016893 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "66d5e98debd4cbe12546394013a546b6e99ad06f",
          "message": "feat: carry the offset convention and the sampling rate in `Events.metadata` (#1827)",
          "timestamp": "2026-11-24T14:13:00+01:00",
          "tree_id": "bce5b780b32ae2789cf128cbe3031911bc8d5eff",
          "url": "https://github.com/pymovements/pymovements/commit/66d5e98debd4cbe12546394013a546b6e99ad06f"
        },
        "date": 1795526550777,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 35.70391054042445,
            "unit": "iter/sec",
            "range": "stddev: 0.004924699480253205",
            "extra": "mean: 28.00813650000009 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 8.763586544239757,
            "unit": "iter/sec",
            "range": "stddev: 0.019189925438109528",
            "extra": "mean: 114.10853250000628 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 16.99534167541984,
            "unit": "iter/sec",
            "range": "stddev: 0.007267318641643941",
            "extra": "mean: 58.83965259999968 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.640170696311144,
            "unit": "iter/sec",
            "range": "stddev: 0.03215900498375835",
            "extra": "mean: 274.7123922000071 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 9.723394304481102,
            "unit": "iter/sec",
            "range": "stddev: 0.03119811679385319",
            "extra": "mean: 102.84474419999015 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.770725853230664,
            "unit": "iter/sec",
            "range": "stddev: 0.011262518089293534",
            "extra": "mean: 102.34654160001355 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.6118977835316433,
            "unit": "iter/sec",
            "range": "stddev: 0.03210942094858944",
            "extra": "mean: 1.634259882799995 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.3645610758277449,
            "unit": "iter/sec",
            "range": "stddev: 0.013837245140226525",
            "extra": "mean: 2.743024602199989 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.1377237076979153,
            "unit": "iter/sec",
            "range": "stddev: 0.02171765397129526",
            "extra": "mean: 7.260914019200027 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.16681948450249745,
            "unit": "iter/sec",
            "range": "stddev: 0.028652599642336976",
            "extra": "mean: 5.99450359760001 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.611184710207674,
            "unit": "iter/sec",
            "range": "stddev: 0.0467065755712258",
            "extra": "mean: 620.6613020000077 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.6043344995680648,
            "unit": "iter/sec",
            "range": "stddev: 0.038819100642030535",
            "extra": "mean: 623.3114105999903 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.5312345134481928,
            "unit": "iter/sec",
            "range": "stddev: 0.03125738901623807",
            "extra": "mean: 653.0678294000154 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 32.91548139118958,
            "unit": "iter/sec",
            "range": "stddev: 0.00436398507317006",
            "extra": "mean: 30.38084079996679 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 47.87242771632658,
            "unit": "iter/sec",
            "range": "stddev: 0.0030327543664293026",
            "extra": "mean: 20.88885079999727 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 40.721063840013336,
            "unit": "iter/sec",
            "range": "stddev: 0.0057003043196412065",
            "extra": "mean: 24.5573152000361 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 49.92307951756793,
            "unit": "iter/sec",
            "range": "stddev: 0.002999312812516157",
            "extra": "mean: 20.030815599989182 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.6241960836633921,
            "unit": "iter/sec",
            "range": "stddev: 0.012338584490351242",
            "extra": "mean: 1.6020606763999923 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 1.2402021349403036,
            "unit": "iter/sec",
            "range": "stddev: 0.016239510362757253",
            "extra": "mean: 806.3201729999719 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.6367070696162715,
            "unit": "iter/sec",
            "range": "stddev: 0.018753801184269957",
            "extra": "mean: 1.5705809589999944 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 1.246423933702403,
            "unit": "iter/sec",
            "range": "stddev: 0.007020826464873627",
            "extra": "mean: 802.2952487999646 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.6211750443165405,
            "unit": "iter/sec",
            "range": "stddev: 0.030901082264987927",
            "extra": "mean: 1.6098521812000173 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 1.230086214535996,
            "unit": "iter/sec",
            "range": "stddev: 0.027630948369427415",
            "extra": "mean: 812.951147800004 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 37.377635759223836,
            "unit": "iter/sec",
            "range": "stddev: 0.0004111205460293908",
            "extra": "mean: 26.753966099988702 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.9196707339952008,
            "unit": "iter/sec",
            "range": "stddev: 0.003574057093349404",
            "extra": "mean: 1.0873456804000228 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.7033140937162807,
            "unit": "iter/sec",
            "range": "stddev: 0.02091517387907919",
            "extra": "mean: 369.9163194999983 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 179.76830992817708,
            "unit": "iter/sec",
            "range": "stddev: 0.0013694165585131407",
            "extra": "mean: 5.5627157000003535 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 131.00998479512126,
            "unit": "iter/sec",
            "range": "stddev: 0.0021412491808628347",
            "extra": "mean: 7.633005999991838 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "66853113+pre-commit-ci[bot]@users.noreply.github.com",
            "name": "pre-commit-ci[bot]",
            "username": "pre-commit-ci[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a75c5122672f8a04fed2f7581a105f81cd614ded",
          "message": "ci: pre-commit autoupdate (#1847)\n\nupdates:\n- [github.com/kynan/nbstripout: 0.9.1 → 0.9.2](https://github.com/kynan/nbstripout/compare/0.9.1...0.9.2)",
          "timestamp": "2026-11-24T13:38:27Z",
          "tree_id": "dcb14f92f25153823571b5eccac0f64a9de7ccbb",
          "url": "https://github.com/pymovements/pymovements/commit/a75c5122672f8a04fed2f7581a105f81cd614ded"
        },
        "date": 1795528256563,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 29.4836542665044,
            "unit": "iter/sec",
            "range": "stddev: 0.0062993587982078745",
            "extra": "mean: 33.91709829999172 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.350597449284184,
            "unit": "iter/sec",
            "range": "stddev: 0.019442051436481703",
            "extra": "mean: 136.04336339998895 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 12.988156282119784,
            "unit": "iter/sec",
            "range": "stddev: 0.010893695978287233",
            "extra": "mean: 76.99322200000438 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.2008899539329603,
            "unit": "iter/sec",
            "range": "stddev: 0.039679334350847945",
            "extra": "mean: 312.4131145999854 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 7.715254520999526,
            "unit": "iter/sec",
            "range": "stddev: 0.036138063619312134",
            "extra": "mean: 129.61335200001258 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 8.084187460613975,
            "unit": "iter/sec",
            "range": "stddev: 0.006014868550404381",
            "extra": "mean: 123.6982696000041 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.48546510560786954,
            "unit": "iter/sec",
            "range": "stddev: 0.02507137235640583",
            "extra": "mean: 2.0598802847999993 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.29351169411730765,
            "unit": "iter/sec",
            "range": "stddev: 0.016332951126322534",
            "extra": "mean: 3.407019277400002 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.10072121296831428,
            "unit": "iter/sec",
            "range": "stddev: 0.02157477280355806",
            "extra": "mean: 9.928395126800035 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.12353241014670721,
            "unit": "iter/sec",
            "range": "stddev: 0.018223030196907186",
            "extra": "mean: 8.095041607400026 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 1.3518378014154666,
            "unit": "iter/sec",
            "range": "stddev: 0.05456009215998178",
            "extra": "mean: 739.7337158000255 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 1.4137635252045715,
            "unit": "iter/sec",
            "range": "stddev: 0.06235633505729249",
            "extra": "mean: 707.3318713999925 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 1.2750637460405676,
            "unit": "iter/sec",
            "range": "stddev: 0.04334453669513786",
            "extra": "mean: 784.2745142000012 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 26.76540001624422,
            "unit": "iter/sec",
            "range": "stddev: 0.003108046492712495",
            "extra": "mean: 37.36166839998987 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 35.90045727989868,
            "unit": "iter/sec",
            "range": "stddev: 0.004724235030563825",
            "extra": "mean: 27.85479840001699 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 35.62946290644464,
            "unit": "iter/sec",
            "range": "stddev: 0.0039027558946941347",
            "extra": "mean: 28.06665939999675 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 40.751306075248436,
            "unit": "iter/sec",
            "range": "stddev: 0.004891141340937574",
            "extra": "mean: 24.539090800021768 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 0.47264155220731036,
            "unit": "iter/sec",
            "range": "stddev: 0.013629181411303827",
            "extra": "mean: 2.1157682715999955 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 0.9530117815570976,
            "unit": "iter/sec",
            "range": "stddev: 0.0052593900208126295",
            "extra": "mean: 1.0493049711999674 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 0.4629659242730488,
            "unit": "iter/sec",
            "range": "stddev: 0.10454662102575549",
            "extra": "mean: 2.159986183800038 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 0.9421875642085713,
            "unit": "iter/sec",
            "range": "stddev: 0.013033448540500791",
            "extra": "mean: 1.061359795000044 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 0.4754356103185317,
            "unit": "iter/sec",
            "range": "stddev: 0.025960844870772572",
            "extra": "mean: 2.103334243999984 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 0.9506137328426758,
            "unit": "iter/sec",
            "range": "stddev: 0.02174613028609047",
            "extra": "mean: 1.051951981599973 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 26.951606176760937,
            "unit": "iter/sec",
            "range": "stddev: 0.0010160001344529713",
            "extra": "mean: 37.103540080006496 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.6268500834950594,
            "unit": "iter/sec",
            "range": "stddev: 0.006306478768261663",
            "extra": "mean: 1.5952777646999892 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.350057033310848,
            "unit": "iter/sec",
            "range": "stddev: 0.0021051243365662105",
            "extra": "mean: 425.5215877000069 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 150.37787101843054,
            "unit": "iter/sec",
            "range": "stddev: 0.002030496838282782",
            "extra": "mean: 6.649914599984186 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 112.73164351743503,
            "unit": "iter/sec",
            "range": "stddev: 0.0026646891172137574",
            "extra": "mean: 8.87062380000998 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6a30dc175aef437ccacba28439276c9dafc8b55b",
          "message": "fix: normalize Duration input per unit (#1859)",
          "timestamp": "2026-11-25T13:50:00Z",
          "tree_id": "607657b6426f60b5f0053560538a7def018ddafc",
          "url": "https://github.com/pymovements/pymovements/commit/6a30dc175aef437ccacba28439276c9dafc8b55b"
        },
        "date": 1795615231167,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 28.852177154268198,
            "unit": "iter/sec",
            "range": "stddev: 0.005796068730622354",
            "extra": "mean: 34.659429500004535 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.227451666557735,
            "unit": "iter/sec",
            "range": "stddev: 0.017614491381923662",
            "extra": "mean: 138.36135420000346 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 13.528832850823928,
            "unit": "iter/sec",
            "range": "stddev: 0.0065169582005922206",
            "extra": "mean: 73.9162063000208 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.1094261853382057,
            "unit": "iter/sec",
            "range": "stddev: 0.03629938119134325",
            "extra": "mean: 321.60274609999533 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 8.151594073321094,
            "unit": "iter/sec",
            "range": "stddev: 0.04413893552496496",
            "extra": "mean: 122.67539220001709 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 8.416401646524005,
            "unit": "iter/sec",
            "range": "stddev: 0.005507027493265739",
            "extra": "mean: 118.81562239998402 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.4919982382168881,
            "unit": "iter/sec",
            "range": "stddev: 0.022698821194977077",
            "extra": "mean: 2.0325276034000126 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.31326046360276805,
            "unit": "iter/sec",
            "range": "stddev: 0.015501966576177912",
            "extra": "mean: 3.1922317566000173 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.10053747566733356,
            "unit": "iter/sec",
            "range": "stddev: 0.055154625145006235",
            "extra": "mean: 9.946539768999969 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.12253895379265047,
            "unit": "iter/sec",
            "range": "stddev: 0.0720089100222894",
            "extra": "mean: 8.160670293400017 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.9246838514733597,
            "unit": "iter/sec",
            "range": "stddev: 0.06334634346059462",
            "extra": "mean: 341.9172979999985 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 3.070677962317454,
            "unit": "iter/sec",
            "range": "stddev: 0.06948377688243382",
            "extra": "mean: 325.6609818000243 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 2.7345328244164224,
            "unit": "iter/sec",
            "range": "stddev: 0.06434340611275281",
            "extra": "mean: 365.69317840001077 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 31.421692448007423,
            "unit": "iter/sec",
            "range": "stddev: 0.007250248856297662",
            "extra": "mean: 31.825147600011405 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 35.023181283352756,
            "unit": "iter/sec",
            "range": "stddev: 0.0048262372707892935",
            "extra": "mean: 28.552517599973726 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 39.84035047266559,
            "unit": "iter/sec",
            "range": "stddev: 0.005667471087452365",
            "extra": "mean: 25.10018080001828 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 39.316888253274215,
            "unit": "iter/sec",
            "range": "stddev: 0.0038546055199396635",
            "extra": "mean: 25.43436279997877 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 3.5954752097916263,
            "unit": "iter/sec",
            "range": "stddev: 0.03044848135389429",
            "extra": "mean: 278.1273521999765 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 5.364333504811172,
            "unit": "iter/sec",
            "range": "stddev: 0.004530938731095525",
            "extra": "mean: 186.41644839999572 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 3.909013412899918,
            "unit": "iter/sec",
            "range": "stddev: 0.011450444538706632",
            "extra": "mean: 255.81902500000524 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 6.828231631222759,
            "unit": "iter/sec",
            "range": "stddev: 0.008545213786196928",
            "extra": "mean: 146.45080220000182 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 3.8864324463739037,
            "unit": "iter/sec",
            "range": "stddev: 0.011881386464566929",
            "extra": "mean: 257.3053858000321 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 6.383574582553944,
            "unit": "iter/sec",
            "range": "stddev: 0.01364255542907737",
            "extra": "mean: 156.65204300000823 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 27.613248745294886,
            "unit": "iter/sec",
            "range": "stddev: 0.0007011140467228454",
            "extra": "mean: 36.21450011999741 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.6543004779915291,
            "unit": "iter/sec",
            "range": "stddev: 0.00488764270371093",
            "extra": "mean: 1.5283497928499856 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.3493317107695244,
            "unit": "iter/sec",
            "range": "stddev: 0.00422362551395676",
            "extra": "mean: 425.6529614000101 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 141.026776472017,
            "unit": "iter/sec",
            "range": "stddev: 0.0019195378866023655",
            "extra": "mean: 7.090851999998904 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 101.82561516235248,
            "unit": "iter/sec",
            "range": "stddev: 0.0028137928650835017",
            "extra": "mean: 9.820711599979859 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8b84103062bb1f988aa41a362d267d7f0863c65a",
          "message": "fix: apply EyeLink message timestamp offsets (#1805)",
          "timestamp": "2026-11-25T14:36:40Z",
          "tree_id": "857c048cfa2e3986f857ccfa3e16d25d3916513b",
          "url": "https://github.com/pymovements/pymovements/commit/8b84103062bb1f988aa41a362d267d7f0863c65a"
        },
        "date": 1795617941775,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 29.548295647657348,
            "unit": "iter/sec",
            "range": "stddev: 0.005867112177866498",
            "extra": "mean: 33.84289950000152 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.241156341906993,
            "unit": "iter/sec",
            "range": "stddev: 0.01274167812333241",
            "extra": "mean: 138.0994903000044 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 13.379149266698619,
            "unit": "iter/sec",
            "range": "stddev: 0.00735736652715397",
            "extra": "mean: 74.74316789999875 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.075502991332935,
            "unit": "iter/sec",
            "range": "stddev: 0.031967927091510195",
            "extra": "mean: 325.15006579999977 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 7.269752828243299,
            "unit": "iter/sec",
            "range": "stddev: 0.028179345110344693",
            "extra": "mean: 137.55625860001146 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 7.399220394757818,
            "unit": "iter/sec",
            "range": "stddev: 0.016812866890184835",
            "extra": "mean: 135.14937339999733 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.7222319724100232,
            "unit": "iter/sec",
            "range": "stddev: 0.016470870448560444",
            "extra": "mean: 1.3845966922000001 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.4555906109670157,
            "unit": "iter/sec",
            "range": "stddev: 0.011084499852342443",
            "extra": "mean: 2.1949530476000065 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.11517455927565975,
            "unit": "iter/sec",
            "range": "stddev: 0.06720475692800706",
            "extra": "mean: 8.6824729896 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.14126729160389975,
            "unit": "iter/sec",
            "range": "stddev: 0.03850739266294653",
            "extra": "mean: 7.078779444600002 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.819351262684599,
            "unit": "iter/sec",
            "range": "stddev: 0.053571378878185534",
            "extra": "mean: 354.69152539999413 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 2.842972922872587,
            "unit": "iter/sec",
            "range": "stddev: 0.050283472068296704",
            "extra": "mean: 351.7444685999976 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 2.477405552256622,
            "unit": "iter/sec",
            "range": "stddev: 0.05125642875067229",
            "extra": "mean: 403.64808219999304 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 33.98611298877901,
            "unit": "iter/sec",
            "range": "stddev: 0.0035473644215429615",
            "extra": "mean: 29.423782599974402 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 35.46765831213951,
            "unit": "iter/sec",
            "range": "stddev: 0.004445774790998384",
            "extra": "mean: 28.194700399990325 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 42.2413621588724,
            "unit": "iter/sec",
            "range": "stddev: 0.002972399849920515",
            "extra": "mean: 23.673479000012776 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 38.06744262631092,
            "unit": "iter/sec",
            "range": "stddev: 0.004697486458334624",
            "extra": "mean: 26.26916680000022 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 4.035453105222367,
            "unit": "iter/sec",
            "range": "stddev: 0.012628537535994392",
            "extra": "mean: 247.80364780001491 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 6.693999095027646,
            "unit": "iter/sec",
            "range": "stddev: 0.010071915892682333",
            "extra": "mean: 149.3875313999979 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 4.099009471819685,
            "unit": "iter/sec",
            "range": "stddev: 0.010124450630238193",
            "extra": "mean: 243.96137820000376 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 6.636089354242845,
            "unit": "iter/sec",
            "range": "stddev: 0.0047602171728629164",
            "extra": "mean: 150.69115959998953 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 4.134765428414445,
            "unit": "iter/sec",
            "range": "stddev: 0.019106715486103907",
            "extra": "mean: 241.85168840000415 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 6.453438152037269,
            "unit": "iter/sec",
            "range": "stddev: 0.008396186324747581",
            "extra": "mean: 154.956160800009 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 35.58329997790393,
            "unit": "iter/sec",
            "range": "stddev: 0.0010181923449013813",
            "extra": "mean: 28.103070839999873 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.8645106494491731,
            "unit": "iter/sec",
            "range": "stddev: 0.007319432797324798",
            "extra": "mean: 1.1567237496000247 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.5422490005783414,
            "unit": "iter/sec",
            "range": "stddev: 0.0033406078212449195",
            "extra": "mean: 393.35250000000315 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 148.32887204652604,
            "unit": "iter/sec",
            "range": "stddev: 0.0016355218177755205",
            "extra": "mean: 6.7417757999692185 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 122.34326701933298,
            "unit": "iter/sec",
            "range": "stddev: 0.00192946110723319",
            "extra": "mean: 8.173723200002314 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df0cf2c6c0a67f42c869825d615eb1daf1980c98",
          "message": "perf: partition events by trial once for events-aware detectors in `Gaze.detect` (#1808)",
          "timestamp": "2026-11-25T14:54:18Z",
          "tree_id": "763e76955fda8ced34597ea98ec7e9303888fd1b",
          "url": "https://github.com/pymovements/pymovements/commit/df0cf2c6c0a67f42c869825d615eb1daf1980c98"
        },
        "date": 1795618990242,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 35.5648282072344,
            "unit": "iter/sec",
            "range": "stddev: 0.005046474170525194",
            "extra": "mean: 28.117667100008248 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 9.14247990010593,
            "unit": "iter/sec",
            "range": "stddev: 0.014294137727341474",
            "extra": "mean: 109.37951309998653 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 17.347669618672633,
            "unit": "iter/sec",
            "range": "stddev: 0.005020445341603923",
            "extra": "mean: 57.64463019999084 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.7964935465633776,
            "unit": "iter/sec",
            "range": "stddev: 0.03112136387729124",
            "extra": "mean: 263.4009482000067 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 9.435054042320118,
            "unit": "iter/sec",
            "range": "stddev: 0.03385189051368379",
            "extra": "mean: 105.98773419999361 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.696359435140742,
            "unit": "iter/sec",
            "range": "stddev: 0.020553399158957427",
            "extra": "mean: 103.1314903999828 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.666708665667933,
            "unit": "iter/sec",
            "range": "stddev: 0.02060729959393668",
            "extra": "mean: 1.4999055081999928 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.42045690090761884,
            "unit": "iter/sec",
            "range": "stddev: 0.0272713849759474",
            "extra": "mean: 2.37836505439999 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.1364243432355859,
            "unit": "iter/sec",
            "range": "stddev: 0.028228103993698912",
            "extra": "mean: 7.330070105400023 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.16384160037413345,
            "unit": "iter/sec",
            "range": "stddev: 0.02289287161722014",
            "extra": "mean: 6.103456007000011 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_after_ivt_benchmark",
            "value": 0.16801033492618375,
            "unit": "iter/sec",
            "range": "stddev: 0.046979231736448766",
            "extra": "mean: 5.952014799799997 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 3.354143357760183,
            "unit": "iter/sec",
            "range": "stddev: 0.0448738726587977",
            "extra": "mean: 298.1387177999977 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 3.5357499950717894,
            "unit": "iter/sec",
            "range": "stddev: 0.05847651874478822",
            "extra": "mean: 282.82542640000656 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 3.0577636791660803,
            "unit": "iter/sec",
            "range": "stddev: 0.04755739169745655",
            "extra": "mean: 327.0363915999951 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 38.22045062111131,
            "unit": "iter/sec",
            "range": "stddev: 0.00476527395311471",
            "extra": "mean: 26.164003399992453 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 45.02177487148124,
            "unit": "iter/sec",
            "range": "stddev: 0.003341240750483927",
            "extra": "mean: 22.211474399989584 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 47.32110224174677,
            "unit": "iter/sec",
            "range": "stddev: 0.005544798491137761",
            "extra": "mean: 21.132221199991363 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 50.257510437093224,
            "unit": "iter/sec",
            "range": "stddev: 0.004095857777263407",
            "extra": "mean: 19.897523600013756 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 4.929686175478071,
            "unit": "iter/sec",
            "range": "stddev: 0.014670539113170924",
            "extra": "mean: 202.8526693999993 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 7.363072873023725,
            "unit": "iter/sec",
            "range": "stddev: 0.019410606950157575",
            "extra": "mean: 135.8128620000116 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 4.746389096212071,
            "unit": "iter/sec",
            "range": "stddev: 0.022016781752340918",
            "extra": "mean: 210.68647760000658 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 7.569270584711348,
            "unit": "iter/sec",
            "range": "stddev: 0.009825704747035233",
            "extra": "mean: 132.1131262000108 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 4.792420871504854,
            "unit": "iter/sec",
            "range": "stddev: 0.00511938438622504",
            "extra": "mean: 208.66280880001113 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 7.3116624549156555,
            "unit": "iter/sec",
            "range": "stddev: 0.010149132796698938",
            "extra": "mean: 136.76780159998998 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 37.7998592812453,
            "unit": "iter/sec",
            "range": "stddev: 0.0005715131525327692",
            "extra": "mean: 26.45512494000627 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.9294185081778554,
            "unit": "iter/sec",
            "range": "stddev: 0.00580060962704581",
            "extra": "mean: 1.0759415604500078 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.896132459909836,
            "unit": "iter/sec",
            "range": "stddev: 0.010282086765381162",
            "extra": "mean: 345.2880743000037 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 176.73550644287167,
            "unit": "iter/sec",
            "range": "stddev: 0.0015027384216164173",
            "extra": "mean: 5.658172599987665 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 135.38847258132708,
            "unit": "iter/sec",
            "range": "stddev: 0.0020245192503552016",
            "extra": "mean: 7.386153200002354 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c6d068ae00ef3ad6288e4e30d139570acaf668da",
          "message": "feat!: detection algorithms take `sampling_rate` and compute durations at the source (#1831)",
          "timestamp": "2026-11-25T15:32:46Z",
          "tree_id": "1bcbf8d243e85347423d4915902a98004b508b71",
          "url": "https://github.com/pymovements/pymovements/commit/c6d068ae00ef3ad6288e4e30d139570acaf668da"
        },
        "date": 1795621260101,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 37.873241170467736,
            "unit": "iter/sec",
            "range": "stddev: 0.004493932838744944",
            "extra": "mean: 26.403866399999742 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 8.382933108690299,
            "unit": "iter/sec",
            "range": "stddev: 0.014185381941932005",
            "extra": "mean: 119.2899892000014 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 16.794525601349292,
            "unit": "iter/sec",
            "range": "stddev: 0.006949682693082531",
            "extra": "mean: 59.543212099998755 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.9978852082771157,
            "unit": "iter/sec",
            "range": "stddev: 0.025209763757587638",
            "extra": "mean: 250.1322444000209 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 9.423447901024568,
            "unit": "iter/sec",
            "range": "stddev: 0.023838864754160454",
            "extra": "mean: 106.11827120000044 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.020200900476922,
            "unit": "iter/sec",
            "range": "stddev: 0.02169593724944089",
            "extra": "mean: 110.86227579999104 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.9272589740327487,
            "unit": "iter/sec",
            "range": "stddev: 0.004583565128957665",
            "extra": "mean: 1.078447368000002 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.5921969332405924,
            "unit": "iter/sec",
            "range": "stddev: 0.006241194099536835",
            "extra": "mean: 1.6886274546000208 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.14449347029993698,
            "unit": "iter/sec",
            "range": "stddev: 0.05119133272582918",
            "extra": "mean: 6.920727960399995 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.17202563803666138,
            "unit": "iter/sec",
            "range": "stddev: 0.044396510236258145",
            "extra": "mean: 5.813086999199993 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_after_ivt_benchmark",
            "value": 0.1724656472672417,
            "unit": "iter/sec",
            "range": "stddev: 0.033676102184906756",
            "extra": "mean: 5.798256150399993 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 3.5135696759306105,
            "unit": "iter/sec",
            "range": "stddev: 0.04369209633085167",
            "extra": "mean: 284.61083519999875 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 3.5156651054992003,
            "unit": "iter/sec",
            "range": "stddev: 0.03788322260298579",
            "extra": "mean: 284.4411995999849 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 3.114018864008748,
            "unit": "iter/sec",
            "range": "stddev: 0.04238587959925436",
            "extra": "mean: 321.12843360000625 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 37.46350296175789,
            "unit": "iter/sec",
            "range": "stddev: 0.006603292179835354",
            "extra": "mean: 26.692645399998582 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 41.925406166393685,
            "unit": "iter/sec",
            "range": "stddev: 0.004238396476682201",
            "extra": "mean: 23.851885800013406 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 40.04959100556756,
            "unit": "iter/sec",
            "range": "stddev: 0.002929515745459344",
            "extra": "mean: 24.969043999999485 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 50.10380958204187,
            "unit": "iter/sec",
            "range": "stddev: 0.002772676225508511",
            "extra": "mean: 19.958562199997232 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 5.2014912321662194,
            "unit": "iter/sec",
            "range": "stddev: 0.012612936283846594",
            "extra": "mean: 192.25255899999638 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 8.542520730093496,
            "unit": "iter/sec",
            "range": "stddev: 0.011848034580422565",
            "extra": "mean: 117.0614659999842 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 5.298167465916492,
            "unit": "iter/sec",
            "range": "stddev: 0.010820553048732442",
            "extra": "mean: 188.7445057999912 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 8.147995311070142,
            "unit": "iter/sec",
            "range": "stddev: 0.008540477067547998",
            "extra": "mean: 122.72957479999604 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 5.2771681763145,
            "unit": "iter/sec",
            "range": "stddev: 0.012523091675712486",
            "extra": "mean: 189.4955715999913 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 8.872798459286232,
            "unit": "iter/sec",
            "range": "stddev: 0.004438409293899792",
            "extra": "mean: 112.70401379999839 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 48.97301779478088,
            "unit": "iter/sec",
            "range": "stddev: 0.0003401397568256598",
            "extra": "mean: 20.41940735999674 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 1.224413619824617,
            "unit": "iter/sec",
            "range": "stddev: 0.002058542132167756",
            "extra": "mean: 816.7174750500067 msec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 3.3063521059699372,
            "unit": "iter/sec",
            "range": "stddev: 0.003985283984544465",
            "extra": "mean: 302.44812650001904 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 153.89457618211702,
            "unit": "iter/sec",
            "range": "stddev: 0.002405781488422386",
            "extra": "mean: 6.497954800022399 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 152.0342679157745,
            "unit": "iter/sec",
            "range": "stddev: 0.0016465849492259422",
            "extra": "mean: 6.577464500003316 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "101206178+itzzdev09@users.noreply.github.com",
            "name": "Devraj Pal",
            "username": "itzzdev09"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2cba579d6b42a7e80ba3c612d63607317595ca57",
          "message": "fix: scale numpy pos2acc by the squared sampling rate (#1840)",
          "timestamp": "2026-11-25T16:52:48+01:00",
          "tree_id": "1e3d872c1b5293dd379d9013347ddd0b022b2cf6",
          "url": "https://github.com/pymovements/pymovements/commit/2cba579d6b42a7e80ba3c612d63607317595ca57"
        },
        "date": 1795622746137,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 28.75297609476719,
            "unit": "iter/sec",
            "range": "stddev: 0.005623369544878499",
            "extra": "mean: 34.77900849999287 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.106307454458758,
            "unit": "iter/sec",
            "range": "stddev: 0.01822776427501612",
            "extra": "mean: 140.7200584000293 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 14.038989730885815,
            "unit": "iter/sec",
            "range": "stddev: 0.008449588830593796",
            "extra": "mean: 71.2301966999803 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.108747737870135,
            "unit": "iter/sec",
            "range": "stddev: 0.04562304992746129",
            "extra": "mean: 321.672932100023 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 7.693864184704419,
            "unit": "iter/sec",
            "range": "stddev: 0.04162965136813412",
            "extra": "mean: 129.9737006000214 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 7.451884149069599,
            "unit": "iter/sec",
            "range": "stddev: 0.03335408306098441",
            "extra": "mean: 134.19424939997953 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.5137947087973674,
            "unit": "iter/sec",
            "range": "stddev: 0.008181062748091937",
            "extra": "mean: 1.9463026435999837 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.33461102680095683,
            "unit": "iter/sec",
            "range": "stddev: 0.02570979289794606",
            "extra": "mean: 2.988544667999986 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.0917152595510467,
            "unit": "iter/sec",
            "range": "stddev: 0.07411292803571423",
            "extra": "mean: 10.903311018199997 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.10711741364918007,
            "unit": "iter/sec",
            "range": "stddev: 0.029158391693844214",
            "extra": "mean: 9.335550270800013 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_after_ivt_benchmark",
            "value": 0.10904090574597859,
            "unit": "iter/sec",
            "range": "stddev: 0.03107581679971153",
            "extra": "mean: 9.170870263400024 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.7878198082899064,
            "unit": "iter/sec",
            "range": "stddev: 0.05359828072036612",
            "extra": "mean: 358.70324079999136 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 2.9040641923594817,
            "unit": "iter/sec",
            "range": "stddev: 0.06557174619124452",
            "extra": "mean: 344.34500540000954 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 2.6310488048821496,
            "unit": "iter/sec",
            "range": "stddev: 0.06383599410944504",
            "extra": "mean: 380.07656799995857 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 35.47803695932115,
            "unit": "iter/sec",
            "range": "stddev: 0.007149403131605413",
            "extra": "mean: 28.186452400018425 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 36.925802024564476,
            "unit": "iter/sec",
            "range": "stddev: 0.0037580229522579047",
            "extra": "mean: 27.081334599984075 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 30.406335921467807,
            "unit": "iter/sec",
            "range": "stddev: 0.0033211443928693415",
            "extra": "mean: 32.88788239999576 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 41.18590089971772,
            "unit": "iter/sec",
            "range": "stddev: 0.004604150007349033",
            "extra": "mean: 24.280153600011545 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 3.927371082583554,
            "unit": "iter/sec",
            "range": "stddev: 0.018265261612973312",
            "extra": "mean: 254.6232527999791 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 6.102796371112304,
            "unit": "iter/sec",
            "range": "stddev: 0.013405345703123933",
            "extra": "mean: 163.85930960002497 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 3.8280340853271455,
            "unit": "iter/sec",
            "range": "stddev: 0.013599293494897656",
            "extra": "mean: 261.2306937999847 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 5.725669378903401,
            "unit": "iter/sec",
            "range": "stddev: 0.008893243614666828",
            "extra": "mean: 174.6520683999961 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 3.954374959511781,
            "unit": "iter/sec",
            "range": "stddev: 0.01645301883387915",
            "extra": "mean: 252.88446600002317 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 6.418854935513077,
            "unit": "iter/sec",
            "range": "stddev: 0.011167261545437346",
            "extra": "mean: 155.7910265999908 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 28.796689705791554,
            "unit": "iter/sec",
            "range": "stddev: 0.0010306716970911525",
            "extra": "mean: 34.72621368000091 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.7031318270859425,
            "unit": "iter/sec",
            "range": "stddev: 0.006405166181657254",
            "extra": "mean: 1.4222084130999975 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.332741736343696,
            "unit": "iter/sec",
            "range": "stddev: 0.006568078845278928",
            "extra": "mean: 428.68011679997835 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 135.32230050629153,
            "unit": "iter/sec",
            "range": "stddev: 0.0019778506437887356",
            "extra": "mean: 7.389764999993531 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 102.59383478340679,
            "unit": "iter/sec",
            "range": "stddev: 0.00269059901985896",
            "extra": "mean: 9.747174400013137 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "daniel@krakowczyk.tech",
            "name": "Daniel G. Krakowczyk",
            "username": "dkrako"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1fab892dbebca676017f0829c359ebece4bd9b85",
          "message": "docs: add Jana Hofmann to core contributors (#1826)",
          "timestamp": "2026-11-25T17:23:34+01:00",
          "tree_id": "0875ea806cfb4b815eef6117b2e22bc563695174",
          "url": "https://github.com/pymovements/pymovements/commit/1fab892dbebca676017f0829c359ebece4bd9b85"
        },
        "date": 1795624623250,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 27.736734114328442,
            "unit": "iter/sec",
            "range": "stddev: 0.005495737729671886",
            "extra": "mean: 36.05327130000546 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 7.231279482531688,
            "unit": "iter/sec",
            "range": "stddev: 0.019823957566314842",
            "extra": "mean: 138.2881138000073 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 13.741191897711566,
            "unit": "iter/sec",
            "range": "stddev: 0.007591878219094135",
            "extra": "mean: 72.77389089999815 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.1579843032605246,
            "unit": "iter/sec",
            "range": "stddev: 0.04203721036597207",
            "extra": "mean: 316.65768539999704 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 7.926966385475859,
            "unit": "iter/sec",
            "range": "stddev: 0.03841307096306406",
            "extra": "mean: 126.15166400001954 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 7.908863158133668,
            "unit": "iter/sec",
            "range": "stddev: 0.016668090218358585",
            "extra": "mean: 126.44042260000104 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.5204087347739288,
            "unit": "iter/sec",
            "range": "stddev: 0.009668517488020308",
            "extra": "mean: 1.9215665172000058 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.33989023851790956,
            "unit": "iter/sec",
            "range": "stddev: 0.011920585213426003",
            "extra": "mean: 2.9421262710000065 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.08488936280398696,
            "unit": "iter/sec",
            "range": "stddev: 0.06275922498531546",
            "extra": "mean: 11.780038946799982 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.09972431954011533,
            "unit": "iter/sec",
            "range": "stddev: 0.0551445328913624",
            "extra": "mean: 10.027644255799988 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_after_ivt_benchmark",
            "value": 0.1018367995021805,
            "unit": "iter/sec",
            "range": "stddev: 0.06342254250617346",
            "extra": "mean: 9.819633029400029 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 2.8711374801378415,
            "unit": "iter/sec",
            "range": "stddev: 0.06189357649754922",
            "extra": "mean: 348.2940148000125 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 3.0584421943473017,
            "unit": "iter/sec",
            "range": "stddev: 0.061497621109238156",
            "extra": "mean: 326.9638385999997 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 2.543939807282378,
            "unit": "iter/sec",
            "range": "stddev: 0.06506031265184961",
            "extra": "mean: 393.09106180003255 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 32.668397076835,
            "unit": "iter/sec",
            "range": "stddev: 0.0063178917037878205",
            "extra": "mean: 30.610623399979886 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 35.751336611972526,
            "unit": "iter/sec",
            "range": "stddev: 0.004927005206291502",
            "extra": "mean: 27.97098220000862 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 36.225884736469865,
            "unit": "iter/sec",
            "range": "stddev: 0.003493254948698471",
            "extra": "mean: 27.604570799985595 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 37.71471458451589,
            "unit": "iter/sec",
            "range": "stddev: 0.004393862796558973",
            "extra": "mean: 26.514849999966827 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 3.8264737676340537,
            "unit": "iter/sec",
            "range": "stddev: 0.006471940042359869",
            "extra": "mean: 261.3372156000196 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 6.629465368714551,
            "unit": "iter/sec",
            "range": "stddev: 0.010060150213354867",
            "extra": "mean: 150.84172619999663 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 3.972907412635046,
            "unit": "iter/sec",
            "range": "stddev: 0.014278964644129675",
            "extra": "mean: 251.70483379997677 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 6.549049218073299,
            "unit": "iter/sec",
            "range": "stddev: 0.011025891352197762",
            "extra": "mean: 152.69392039997456 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 3.8361914988062833,
            "unit": "iter/sec",
            "range": "stddev: 0.012199593641680924",
            "extra": "mean: 260.67520360002163 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 6.430094107925454,
            "unit": "iter/sec",
            "range": "stddev: 0.012143847396285946",
            "extra": "mean: 155.51871919999485 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 26.684634605274532,
            "unit": "iter/sec",
            "range": "stddev: 0.0008825572595333397",
            "extra": "mean: 37.474749599994084 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.6384488903413715,
            "unit": "iter/sec",
            "range": "stddev: 0.00500229886833602",
            "extra": "mean: 1.566296089049996 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.347576146913567,
            "unit": "iter/sec",
            "range": "stddev: 0.007338383299080699",
            "extra": "mean: 425.97127310001497 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 143.8196621094405,
            "unit": "iter/sec",
            "range": "stddev: 0.0017325011685566198",
            "extra": "mean: 6.953152200003387 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 102.84717173575865,
            "unit": "iter/sec",
            "range": "stddev: 0.0027650800309886297",
            "extra": "mean: 9.723164799993356 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71385469+kirthan98@users.noreply.github.com",
            "name": "kirthan98",
            "username": "kirthan98"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4c6ad9d6a3ae2003374cad6bc8ca57ed00bf8889",
          "message": "feat: add statistical sample generator (#1822)",
          "timestamp": "2026-11-25T16:44:26Z",
          "tree_id": "ca11b331b18e1c5ed9363daf6a983cd09508ba3b",
          "url": "https://github.com/pymovements/pymovements/commit/4c6ad9d6a3ae2003374cad6bc8ca57ed00bf8889"
        },
        "date": 1795625711630,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-100]",
            "value": 34.518218114907576,
            "unit": "iter/sec",
            "range": "stddev: 0.004698614546858678",
            "extra": "mean: 28.970209199997043 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[False-400]",
            "value": 8.175055544740086,
            "unit": "iter/sec",
            "range": "stddev: 0.014918413232926938",
            "extra": "mean: 122.32332790000555 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-100]",
            "value": 14.790560246174358,
            "unit": "iter/sec",
            "range": "stddev: 0.007395004832128467",
            "extra": "mean: 67.61069109999767 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/event_properties_test.py::test_event_properties_benchmark[True-400]",
            "value": 3.470410897889266,
            "unit": "iter/sec",
            "range": "stddev: 0.03199886353901949",
            "extra": "mean: 288.1503168999984 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_benchmark",
            "value": 8.997648496956977,
            "unit": "iter/sec",
            "range": "stddev: 0.03253224575970055",
            "extra": "mean: 111.1401495999985 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/events_map_to_aois_test.py::test_events_map_to_aois_keyed_benchmark",
            "value": 9.301817849238269,
            "unit": "iter/sec",
            "range": "stddev: 0.018259473037884527",
            "extra": "mean: 107.50586780001186 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[monocular]",
            "value": 0.6647872631703177,
            "unit": "iter/sec",
            "range": "stddev: 0.02917443337184769",
            "extra": "mean: 1.504240612600006 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/from_asc_test.py::test_from_asc_benchmark[binocular]",
            "value": 0.41891781619249485,
            "unit": "iter/sec",
            "range": "stddev: 0.01580821750031949",
            "extra": "mean: 2.387103057799993 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_benchmark",
            "value": 0.11340029610597246,
            "unit": "iter/sec",
            "range": "stddev: 0.05492225941131566",
            "extra": "mean: 8.818319125599999 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_empty_events_benchmark",
            "value": 0.13547553676496688,
            "unit": "iter/sec",
            "range": "stddev: 0.04909584204152343",
            "extra": "mean: 7.381406443399999 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_detect_test.py::test_gaze_detect_fill_after_ivt_benchmark",
            "value": 0.13786889672759323,
            "unit": "iter/sec",
            "range": "stddev: 0.11504941573999375",
            "extra": "mean: 7.2532675878000195 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[list]",
            "value": 3.389501083438768,
            "unit": "iter/sec",
            "range": "stddev: 0.044584429171466086",
            "extra": "mean: 295.02867100000003 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_direct]",
            "value": 3.498467719767735,
            "unit": "iter/sec",
            "range": "stddev: 0.05559345979043151",
            "extra": "mean: 285.83942459997616 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_map_to_aois_test.py::test_gaze_map_to_aois_benchmark[flat_average_lr]",
            "value": 3.167943435411967,
            "unit": "iter/sec",
            "range": "stddev: 0.05570321311276125",
            "extra": "mean: 315.66220180000073 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-location]",
            "value": 47.232827487788406,
            "unit": "iter/sec",
            "range": "stddev: 0.003211788082538909",
            "extra": "mean: 21.17171579996011 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[many_short_trials-data_loss]",
            "value": 46.72249063618136,
            "unit": "iter/sec",
            "range": "stddev: 0.003965629224650884",
            "extra": "mean: 21.40296860000035 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-location]",
            "value": 47.88126778595076,
            "unit": "iter/sec",
            "range": "stddev: 0.00439767328734286",
            "extra": "mean: 20.884994200037 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_measure_samples_test.py::test_gaze_measure_samples_benchmark[few_long_trials-data_loss]",
            "value": 46.376064845476535,
            "unit": "iter/sec",
            "range": "stddev: 0.003609848504282673",
            "extra": "mean: 21.562847199993485 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-pix2deg]",
            "value": 4.802177722286574,
            "unit": "iter/sec",
            "range": "stddev: 0.015923844844886268",
            "extra": "mean: 208.23885700003757 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[1_trial-deg2pix]",
            "value": 7.781060237059267,
            "unit": "iter/sec",
            "range": "stddev: 0.011050303135343722",
            "extra": "mean: 128.51719039999807 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-pix2deg]",
            "value": 4.720545749771555,
            "unit": "iter/sec",
            "range": "stddev: 0.012417820894443748",
            "extra": "mean: 211.83991279999645 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[200_trials-deg2pix]",
            "value": 7.819039380281147,
            "unit": "iter/sec",
            "range": "stddev: 0.013283131979026367",
            "extra": "mean: 127.8929484000173 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-pix2deg]",
            "value": 4.882291877949589,
            "unit": "iter/sec",
            "range": "stddev: 0.009636295623189935",
            "extra": "mean: 204.8218388000123 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/gaze_transform_test.py::test_gaze_element_wise_transform_benchmark[2660_trials-deg2pix]",
            "value": 7.969374573332617,
            "unit": "iter/sec",
            "range": "stddev: 0.014456118147019196",
            "extra": "mean: 125.48036120001599 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_viterbi_only",
            "value": 37.27607153004809,
            "unit": "iter/sec",
            "range": "stddev: 0.0007697239146919594",
            "extra": "mean: 26.82686128000114 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmark/ihmm_test.py::test_ihmm_benchmark_with_reestimation",
            "value": 0.9276414275620053,
            "unit": "iter/sec",
            "range": "stddev: 0.011030143751261167",
            "extra": "mean: 1.0780027392999956 sec\nrounds: 20"
          },
          {
            "name": "tests/benchmark/import_test.py::test_import_pymovements_subprocess",
            "value": 2.916892224108582,
            "unit": "iter/sec",
            "range": "stddev: 0.011026754548223898",
            "extra": "mean: 342.8306303999989 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_null_ratio_benchmark",
            "value": 171.7241253297969,
            "unit": "iter/sec",
            "range": "stddev: 0.001612092455301844",
            "extra": "mean: 5.823293600008128 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmark/sample_measures_test.py::test_data_loss_benchmark",
            "value": 125.72030347818594,
            "unit": "iter/sec",
            "range": "stddev: 0.003053357429276242",
            "extra": "mean: 7.954164700004185 msec\nrounds: 10"
          }
        ]
      }
    ]
  }
}