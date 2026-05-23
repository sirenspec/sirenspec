def process_data(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
        else:
            results.append(0)
    for i in range(len(results)):
        results[i] = results[i] + 1
    total = 0
    for r in results:
        total = total + r
    return results, total


def validate(x, y, z, a, b, c, d):
    if x and y and z:
        if a or b:
            if c and d:
                return True
    return False


class DataProcessor:
    def run(self, data):
        return process_data(data)

    def check(self, x, y, z, a, b, c, d):
        return validate(x, y, z, a, b, c, d)
