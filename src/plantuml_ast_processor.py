import re
from collections import OrderedDict
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError

from src.name_formatter import NameFormatter


class PlantUMLASTProcessor:
    # ----------------- Method 1 -----------------
    @staticmethod
    def parse_plantuml_to_ast(plantuml_code):
        primitive_types = {"string", "int", "double", "float", "boolean", "char", "long", "short", "byte"}
        class_pattern = r'class\s+(\w+)\s*\{([^}]*)\}'
        relation_pattern = r'''
            (?P<class1>\w+)
            (?:\s+"(?P<multiplicity1>[^"]+)")?
            \s*
            (?P<relation>
                <\|-- | <\|\.\. |
                \*-- | o-- |
                \.\.> | --> |
                <--\* | \.\.\|> | <-- |
                -- | \.\. | <-\| | \|-> | --\|> | <\|--
            )
            \s*
            (?:"(?P<multiplicity2>[^"]+)"\s+)?        
            (?P<class2>\w+)
            (?:\s*:\s*(?P<label>.+))?
            $
        '''

        member_patterns = [
            re.compile(r'^\s*([+\-#~]?)\s*([\w_]+(?:\([^)]*\))?)\s*(?::\s*([\w<>]+))?\s*$'),
            re.compile(r'^\s*([\w_]+)\s*$')
        ]

        ast = OrderedDict([
            ("type", "root"),
            ("children", [])
        ])

        class_map = {}

        class_matches = re.findall(class_pattern, plantuml_code, re.DOTALL)
        for class_name, class_body in class_matches:
            class_name = NameFormatter.to_pascal_case(class_name)
            if class_name not in class_map:
                class_node = OrderedDict([
                    ("type", "class"),
                    ("value", class_name),
                    ("children", [])
                ])
                class_map[class_name] = class_node
                ast["children"].append(class_node)
            else:
                class_node = class_map[class_name]

            for line in class_body.strip().splitlines():
                line = line.strip()
                for pattern in member_patterns:
                    member_match = pattern.match(line)
                    if member_match:
                        if pattern == member_patterns[0]:
                            visibility = member_match.group(1) or ""
                            name = member_match.group(2)
                            data_type = member_match.group(3) or None
                            is_method = '(' in name
                        else:
                            visibility = ""
                            name = member_match.group(1)
                            is_method = False
                            data_type = None

                        if not any(child["value"] == name for child in class_node["children"]):
                            member_node = OrderedDict([
                                ("type", "method" if is_method else "attribute"),
                                ("value", name),
                                ("visibility", visibility),
                            ])
                            if data_type:
                                if data_type.lower() in primitive_types:
                                    member_node["data_type"] = data_type.lower()
                                else:
                                    member_node["data_type"] = data_type

                            class_node["children"].append(member_node)
                        break

        relation_matches = re.finditer(relation_pattern, plantuml_code, re.VERBOSE | re.MULTILINE)
        for match in relation_matches:
            data = match.groupdict()
            source_class = data["class1"]
            target_class = data["class2"]
            relation_type = data["relation"].strip()
            label = (data.get("label") or "").strip()
            multiplicity1 = data.get("multiplicity1")
            multiplicity2 = data.get("multiplicity2")

            for cls in [source_class, target_class]:
                if cls not in class_map:
                    class_node = OrderedDict([
                        ("type", "class"),
                        ("value", cls),
                        ("children", [])
                    ])
                    class_map[cls] = class_node
                    ast["children"].append(class_node)

            relation_node = OrderedDict([
                ("type", "relation"),
                ("value", relation_type),
                ("children", [OrderedDict([
                    ("type", "class"),
                    ("value", target_class)
                ])])
            ])
            if label:
                relation_node["label"] = label
            if multiplicity1:
                relation_node["multiplicity1"] = multiplicity1
            if multiplicity2:
                relation_node["multiplicity2"] = multiplicity2

            class_map[source_class]["children"].append(relation_node)

        return ast

    # ----------------- Method 2 -----------------
    @staticmethod
    def generate_plantuml_from_ast(ast):
        if isinstance(ast, str):
            try:
                ast = json.loads(ast)
            except json.JSONDecodeError:
                raise ValueError("Input must be valid JSON or AST dictionary")

        plantuml_lines = []

        if not isinstance(ast, dict) or "children" not in ast:
            raise ValueError("Invalid AST structure - missing 'children'")

        class_nodes = [node for node in ast["children"] if isinstance(node, dict) and node.get("type") == "class"]

        for class_node in class_nodes:
            if not isinstance(class_node, dict) or "value" not in class_node:
                continue

            class_lines = []
            class_lines.append(f"class {class_node['value']} {{")

            for member in class_node.get("children", []):
                if isinstance(member, dict) and member.get("type") in ("attribute", "method"):
                    line = f"  {member.get('visibility', '')} {member['value']}"
                    if member.get("data_type"):
                        line += f": {member['data_type']}"
                    class_lines.append(line)

            class_lines.append("}")
            plantuml_lines.extend(class_lines)

        for class_node in class_nodes:
            for relation in [child for child in class_node.get("children", [])
                            if isinstance(child, dict) and child.get("type") == "relation"]:
                if not relation.get("children"):
                    continue

                source = class_node['value']
                target = relation['children'][0].get('value', '')
                if not target:
                    continue

                line = f"{source} {relation.get('value', '')} {target}"

                if relation.get("label"):
                    line += f" : {relation['label']}"
                if relation.get("multiplicity1"):
                    line = line.replace(source, f'{source} "{relation["multiplicity1"]}"', 1)
                if relation.get("multiplicity2"):
                    line = line.replace(target, f'"{relation["multiplicity2"]}" {target}', 1)

                plantuml_lines.append(line)

        return "\n".join(plantuml_lines)

    # ----------------- Method 3 -----------------
    @staticmethod
    def validate_ast_json(input_json_str, schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)

        try:
            data = json.loads(input_json_str)
            validate(instance=data, schema=schema)
            return True,"Valid"
        except json.JSONDecodeError as e:
            return False, f"❌ Invalid JSON format: {e}"
        except ValidationError as e:
            return False, f"❌ JSON validation error: {e.message}"
        except SchemaError as e:

            return False, f"❌ Schema error: {e}"

    @staticmethod
    def json_ast_to_tokens(ast: dict) -> str:
        tokens = ["[AST:SEG]"]
        tokens.append("\n")


        def split_camel_case(name):
            parts = re.sub('([a-z])([A-Z])', r'\1 \2', name).split()
            return ' '.join([p.lower() for p in parts])

        def manipulate_string(name):
            return name

        def parse_parameters(signature):
            params = []
            match = re.match(r"(.*?)\((.*?)\)", signature)
            if not match:
                return manipulate_string(signature), []
            method_name, param_str = match.groups()
            method_name = manipulate_string(method_name)
            param_parts = [p.strip() for p in param_str.split(',') if p.strip()]
            for part in param_parts:
                if ':' in part:
                    name, typ = map(str.strip, part.split(':'))
                    params.append((name, typ))
            return method_name, params

        def traverse(node):
            if node["type"] == "class":
                tokens.append("[CLASS]")
                tokens.append(NameFormatter.uppercase_first_letter(node["value"]))
                tokens.append("\n")
                for child in node.get("children", []):
                    traverse(child)

            elif node["type"] == "attribute":
                tokens.append("[ATTRIBUTE]")
                visibility = node.get("visibility", "")
                if visibility == "+" or visibility == "":
                    tokens.append("[PUBLIC]")
                elif visibility == "-":
                    tokens.append("[PRIVATE]")
                elif visibility == "#":
                    tokens.append("[PROTECTED]")
                elif visibility == "~":
                    tokens.append("[PACKAGE]")
                tokens.append(NameFormatter.to_camel_case(node["value"]))
                tokens.append("[TYPE]")
                if "data_type" in node and node["data_type"]:
                    tokens.append(node["data_type"])
                else:
                    tokens.append("string")
                tokens.append("\n")

            elif node["type"] == "method":
                tokens.append("[METHOD]")
                visibility = node.get("visibility", "")
                if visibility == "+" or visibility == "":
                    tokens.append("[PUBLIC]")
                elif visibility == "-":
                    tokens.append("[PRIVATE]")
                elif visibility == "#":
                    tokens.append("[PROTECTED]")
                elif visibility == "~":
                    tokens.append("[PACKAGE]")

                method_name, params = parse_parameters(node["value"])
                tokens.append(NameFormatter.to_camel_case(method_name))
                for param_name, param_type in params:
                    tokens.append("[PARAM]")
                    tokens.append(manipulate_string(param_name))
                    tokens.append("[PARAMTYPE]")
                    tokens.append(param_type)

                tokens.append("[TYPE]")
                if "data_type" in node and node["data_type"]:
                    tokens.append(node["data_type"])
                else:
                    tokens.append("void")
                tokens.append("\n")

            elif node["type"] == "relation":
                return
                # relation type
                rel_type = node["value"]
                rel_map = {
                    "--": "[ASSOCIATION]",

                    "<|--": "[RIGHT:EXTENSION]",
                    "--|>": "[LEFT:EXTENSION]",

                    "<|..": "[RIGHT:IMPLEMENTATION]",
                    "..|>": "[LEFT:IMPLEMENTATION]",

                    "*--": "[LEFT:COMPOSITION]",
                    "--*": "[RIGHT:COMPOSITION]",

                    "o--": "[LEFT:AGGREGATION]",
                    "--o": "[RIGHT:AGGREGATION]",

                    "-->": "[LEFT:ASSOCIATION]",
                    "<--": "[RIGHT:ASSOCIATION]",

                    "..": "[DEPENDENCY]",
                    "..>": "[LEFT:DEPENDENCY]",
                    "<..": "[RIGHT:DEPENDENCY]"
                }
                token = rel_map.get(rel_type)
                if token:
                    tokens.append(token)
                else:
                    raise ValueError(f"Unknown relation type: {rel_type}")

                # multiplicity1
                mult1 = node.get("multiplicity1")
                if mult1 == "1":
                    tokens.append("[ONE]")
                elif mult1 == "*":
                    tokens.append("[MANY]")
                elif mult1 == "0..*":
                    tokens.append("[MOM]")
                elif mult1 == "1..*":
                    tokens.append("[OOM]")
                elif mult1 == "0..1":
                    tokens.append("[ZOO]")

                # multiplicity2
                mult2 = node.get("multiplicity2")
                if mult2 == "1":
                    tokens.append("[ONE]")
                elif mult2 == "*":
                    tokens.append("[MANY]")
                elif mult2 == "0..*":
                    tokens.append("[MOM]")
                elif mult2 == "1..*":
                    tokens.append("[OOM]")
                elif mult2 == "0..1":
                    tokens.append("[ZOO]")

                # target class
                if node.get("children"):
                    related_class = node["children"][0]
                    tokens.append("[TARGET]")
                    tokens.append(manipulate_string(related_class["value"]))

                tokens.append("[LABEL]")
                # label
                if "label" in node and node["label"]:
                    label = node["label"].replace("_"," ").strip('"')
                    tokens.append(NameFormatter.to_camel_case(label))
                else:
                    tokens.append("noLabel")
                tokens.append("\n")

            elif node["type"] == "root":
                for child in node.get("children", []):
                    traverse(child)
            else:
                raise ValueError(f"Unknown node type: {node['type']}")

        traverse(ast)
        tokens.append("[/AST:SEG]")
        return " ".join(tokens)

    @staticmethod
    def validate_all_ast_json_file(file_path_to_validate):
        # Read and parse the JSON file
        with open(file_path_to_validate, 'r') as f:
            data = json.load(f)

        for i, item in enumerate(data, 1):
            if 'PlantUML' in item:
                ast = item['Output_AST']
                isValidated = PlantUMLASTProcessor.validate_ast_json(json.dumps(ast, indent=4), schema_path)
                if not isValidated:
                    print(i)
                    print(item['Model'])
                    print(item['PlantUML'])
                    print(json.dumps(ast, indent=4))















