import wordninja
import re
class NameFormatter:

    @staticmethod
    def to_camel_case_attr(name: str) -> str:
        if not name:
            return ""

        # First try splitting by delimiters (_ - space)
        parts = re.split(r'[\s_-]+', name.strip())

        # If splitting fails (i.e., only one part and all lowercase), use wordninja
        if len(parts) == 1:
            parts = wordninja.split(name.lower())

        if not parts:
            return ""

        return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

    @staticmethod
    def to_camel_case_method(name: str) -> str:
        if not name:
            return ""

        if '(' in name:
            method_name, rest = name.split('(', 1)

            camel_name = NameFormatter.to_camel_case_attr(method_name.strip())
            return f"{camel_name}({rest}"
        else:
            # No parentheses, treat whole string as method name
            return NameFormatter.to_camel_case_attr(name.strip())


    @staticmethod
    def to_pascal_case(name: str) -> str:
        if not name:
            return ""

        # First split on delimiters like space, underscore, dash
        parts = re.split(r'[\s_-]+', name.strip())

        # If splitting didn't produce multiple parts, use wordninja
        if len(parts) == 1:
            parts = wordninja.split(name)

        return ''.join(word.capitalize() for word in parts if word)


    @staticmethod
    def lowercase_first_letter(s: str) -> str:
        return s[:1].lower() + s[1:] if s else ""

    @staticmethod
    def uppercase_first_letter(s: str) -> str:
        return s[:1].upper() + s[1:] if s else ""