import random
class ASTClassDescriptor:
    @staticmethod
    def ast_to_nl(ast):
        if ast["type"] != "root":
            return ""

        descriptions = []

        class_templates = [
            "The {class_name} has {features}.",
            "{class_name} is defined by {features}.",
            "{class_name} includes {features}.",
            "Class {class_name} contains {features}.",
            "In {class_name}, you can find {features}.",
            "{class_name} encapsulates {features}.",
            "Among the features of {class_name} are {features}.",
            "The design of {class_name} integrates {features}.",
            "The structure of {class_name} comprises {features}.",
        ]

        attr_templates = [
            "{item}",
            "an attribute {item}",
            "{item} attribute"
        ]

        method_templates = [
            "a method {item}",
            "method {item}",
            "{item}"
        ]

        relation_template1 = {
            "-->": "depends on",
            "<|--": "inherits from",
            "*--": "is composed of",
            "o--": "is aggregated in",
            "--": "is associated with",
            "<|..": "implements",
            "..>": "uses",
        }

        relation_template2 = {
            "-->": "needs",
            "<|--": "comes from",
            "*--": "has as a part",
            "o--": "includes",
            "--": "is linked to",
            "<|..": "does what is defined by",
            "..>": "uses"
        }

        relation_templates = [relation_template1, relation_template2]

        def describe_class(cls):
            class_name = cls["value"]
            attributes = []
            methods = []
            relations = []

            for child in cls.get("children", []):
                if child["type"] == "attribute":
                    attr_str = child["value"]
                    attributes.append(random.choice(attr_templates).format(item=attr_str))
                elif child["type"] == "method":
                    method_str = child["value"].split('(')[0].strip()  # Only method name
                    methods.append(random.choice(method_templates).format(item=method_str))
                elif child["type"] == "relation":
                    relation_type = child["value"]
                    target_class = child["children"][0]["value"]
                    label = random.choice(relation_templates).get(relation_type, "is related to")
                    if relation_type == "<|--":
                        relations.append(f"{class_name} is a subclass of {target_class}.")
                    else:
                        relations.append(f"{class_name} {label} {target_class}.")

            feature_parts = []

            if attributes:
                feature_parts.append(", ".join(attributes))
            if methods:
                feature_parts.append(", ".join(methods))

            # Shuffle features before formatting
            random.shuffle(feature_parts)

            if feature_parts:
                features_text = " and ".join(feature_parts)
                descriptions.append(
                    random.choice(class_templates).format(class_name=class_name, features=features_text))
            if relations:
                descriptions.extend(relations)

        for cls in ast.get("children", []):
            if cls["type"] == "class":
                describe_class(cls)

        return " ".join(descriptions)
