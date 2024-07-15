class ApiDoc:
    @staticmethod
    def list_routes(app):
        output = []
        for rule in app.url_map.iter_rules():
            methods = ','.join(rule.methods)
            line = f"{rule.endpoint}: {rule.rule} [{methods}]"
            output.append(line)

        for line in sorted(output):
            print(line)