window.BENCHMARK_DATA = {
  "lastUpdate": 1794939066177,
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
      }
    ]
  }
}