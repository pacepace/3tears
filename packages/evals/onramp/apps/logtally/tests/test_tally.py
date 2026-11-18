import unittest

from tally import day_files, parse, tally, tally_file

UA = '"-" "Mozilla/5.0 (X11; Linux x86_64)"'


def line(target="/menu", status=200, size="5120", method="GET"):
    return f'203.0.113.9 - - [02/Nov/2026:08:14:07 +0000] "{method} {target} HTTP/1.1" {status} {size} {UA}'


class ParseTests(unittest.TestCase):
    def test_a_request(self):
        request = parse(line())
        self.assertEqual((request.method, request.page, request.status, request.size), ("GET", "/menu", 200, 5120))

    def test_the_query_string_is_not_part_of_the_page(self):
        self.assertEqual(parse(line("/menu?lang=en")).page, "/menu")

    def test_a_line_not_in_the_format(self):
        self.assertIsNone(parse('203.0.113.9 - - [02/Nov/2026:08:14:07 +0000] "GET /me'))
        self.assertIsNone(parse("trace: cache warmed"))


class TallyTests(unittest.TestCase):
    def test_hits_and_bytes_by_page(self):
        result = tally([line(), line("/menu?lang=en", size="5000"), line("/about", size="610")])
        self.assertEqual((result.pages["/menu"].hits, result.pages["/menu"].bytes), (2, 10120))
        self.assertEqual(result.pages["/about"].hits, 1)

    def test_not_found_and_errors(self):
        result = tally([line("/specials", 404, "512"), line("/order", 502, "157"), line("/order", 200, "9000")])
        self.assertEqual(result.pages["/specials"].not_found, 1)
        self.assertEqual((result.pages["/order"].errors, result.pages["/order"].hits), (1, 2))

    def test_unparsed_lines_are_counted_and_blank_lines_skipped(self):
        result = tally([line(), "", "garbage", line()[:40]])
        self.assertEqual((result.unparsed, result.pages["/menu"].hits), (2, 1))

    def test_the_busiest_pages(self):
        result = tally([line("/b"), line("/a"), line("/c"), line("/c")])
        self.assertEqual(result.busiest(2), [("/c", 2), ("/a", 1)])


class LogFileTests(unittest.TestCase):
    def test_the_day_files_tally(self):
        files = day_files()
        self.assertEqual(len(files), 4)
        for path in files:
            result = tally_file(path)
            self.assertGreater(sum(stats.hits for stats in result.pages.values()), 50)
            self.assertIn("/", result.pages)


if __name__ == "__main__":
    unittest.main()
