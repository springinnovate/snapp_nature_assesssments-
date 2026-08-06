"""Parse and evaluate the constrained PAD-US land-classification rule DSL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Mapping


PADUS_RULE_FIELDS = ("Own_Type", "Mang_Type", "Des_Tp")
VIRTUAL_MANAGER_FIELD = "manager"
ALLOWED_FIELDS = frozenset((*PADUS_RULE_FIELDS, VIRTUAL_MANAGER_FIELD))
REQUIRED_RULES = frozenset(("public_land", "public_access"))
NASA_LOCAL_MANAGER = "National Aeronautics and Space Administration (NASA)"
VIRTUAL_MANAGER_VALUES = frozenset(("NASA", "DOE"))


class RuleConfigError(ValueError):
    """Raised when the PAD-US rule configuration is invalid."""


@dataclass(frozen=True)
class Membership:
    """Test whether one configured field value belongs to a set."""

    field: str
    values: frozenset[str]
    negate: bool = False


@dataclass(frozen=True)
class Reference:
    """Reference another named Boolean rule."""

    name: str


@dataclass(frozen=True)
class BooleanExpression:
    """Combine two expressions with `AND` or `OR`."""

    operator: str
    left: Expression
    right: Expression


Expression = Membership | Reference | BooleanExpression


@dataclass(frozen=True)
class RuleSet:
    """Named PAD-US classification expressions."""

    rules: Mapping[str, Expression]

    def evaluate(self, attributes: Mapping[str, object]) -> dict[str, bool]:
        """Evaluate all configured rules against one feature's attributes."""
        results: dict[str, bool] = {}
        active_rules: set[str] = set()

        def evaluate_rule(name: str) -> bool:
            if name in results:
                return results[name]
            if name in active_rules:
                raise RuleConfigError(f"Rule reference cycle includes {name!r}.")
            if name not in self.rules:
                raise RuleConfigError(f"Unknown rule reference: {name!r}.")

            active_rules.add(name)
            result = _evaluate_expression(
                self.rules[name],
                attributes,
                evaluate_rule,
            )
            active_rules.remove(name)
            results[name] = result
            return result

        for name in self.rules:
            evaluate_rule(name)
        return results

    def validate(
        self,
        allowed_values: Mapping[str, set[str] | frozenset[str]],
    ) -> None:
        """Validate required rules, references, fields, and domain codes."""
        missing_rules = REQUIRED_RULES.difference(self.rules)
        extra_rules = set(self.rules).difference(REQUIRED_RULES)
        if missing_rules:
            raise RuleConfigError(
                "Missing required rule(s): " + ", ".join(sorted(missing_rules))
            )
        if extra_rules:
            raise RuleConfigError(
                "Unsupported rule name(s): " + ", ".join(sorted(extra_rules))
            )

        for rule_name, expression in self.rules.items():
            for item in _walk_expression(expression):
                if isinstance(item, Reference):
                    if item.name not in self.rules:
                        raise RuleConfigError(
                            f"Rule {rule_name!r} references unknown rule "
                            f"{item.name!r}."
                        )
                    continue
                if not isinstance(item, Membership):
                    continue
                if item.field not in ALLOWED_FIELDS:
                    raise RuleConfigError(
                        f"Rule {rule_name!r} uses unsupported field "
                        f"{item.field!r}."
                    )
                field_values = allowed_values.get(item.field)
                if field_values is None:
                    raise RuleConfigError(
                        f"No allowed values were supplied for {item.field!r}."
                    )
                unknown_values = item.values.difference(field_values)
                if unknown_values:
                    raise RuleConfigError(
                        f"Rule {rule_name!r} has invalid {item.field} value(s): "
                        + ", ".join(sorted(unknown_values))
                    )

        # Evaluate once to detect reference cycles before processing features.
        self.evaluate({field: None for field in ALLOWED_FIELDS})


@dataclass(frozen=True)
class Token:
    """One token in the constrained rule language."""

    value: str
    offset: int


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[=(){},]")


class _Parser:
    """Recursive-descent parser for the small PAD-US rule language."""

    def __init__(self, text: str) -> None:
        self.tokens = _tokenize(text)
        self.index = 0

    def parse(self) -> RuleSet:
        """Parse all `name = expression` assignments."""
        rules: dict[str, Expression] = {}
        while not self._at_end():
            name = self._take_word("rule name")
            if name in rules:
                raise self._error(f"Duplicate rule name {name!r}.")
            self._expect("=")
            rules[name] = self._parse_or()
        return RuleSet(rules)

    def _parse_or(self) -> Expression:
        expression = self._parse_and()
        while self._match_keyword("OR"):
            expression = BooleanExpression("OR", expression, self._parse_and())
        return expression

    def _parse_and(self) -> Expression:
        expression = self._parse_primary()
        while self._match_keyword("AND"):
            expression = BooleanExpression("AND", expression, self._parse_primary())
        return expression

    def _parse_primary(self) -> Expression:
        if self._match("("):
            expression = self._parse_or()
            self._expect(")")
            return expression

        identifier = self._take_word("field or rule name")
        negate = self._match_keyword("NOT")
        if negate or self._peek_keyword("IN"):
            self._expect_keyword("IN")
            return Membership(identifier, self._parse_set(), negate=negate)
        return Reference(identifier)

    def _parse_set(self) -> frozenset[str]:
        self._expect("{")
        values = [self._take_word("set value")]
        while self._match(","):
            values.append(self._take_word("set value"))
        self._expect("}")
        if len(values) != len(set(values)):
            raise self._error("A rule set contains a duplicate value.")
        return frozenset(values)

    def _at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def _peek(self) -> Token | None:
        if self._at_end():
            return None
        return self.tokens[self.index]

    def _match(self, value: str) -> bool:
        token = self._peek()
        if token is None or token.value != value:
            return False
        self.index += 1
        return True

    def _match_keyword(self, value: str) -> bool:
        if not self._peek_keyword(value):
            return False
        self.index += 1
        return True

    def _peek_keyword(self, value: str) -> bool:
        token = self._peek()
        return token is not None and token.value.upper() == value

    def _expect(self, value: str) -> None:
        if not self._match(value):
            raise self._error(f"Expected {value!r}.")

    def _expect_keyword(self, value: str) -> None:
        if not self._match_keyword(value):
            raise self._error(f"Expected {value!r}.")

    def _take_word(self, description: str) -> str:
        token = self._peek()
        if token is None or not token.value[0].isalpha() and token.value[0] != "_":
            raise self._error(f"Expected {description}.")
        self.index += 1
        return token.value

    def _error(self, message: str) -> RuleConfigError:
        token = self._peek()
        if token is None:
            return RuleConfigError(f"{message} Reached end of configuration.")
        return RuleConfigError(f"{message} Near offset {token.offset}.")


def parse_rule_config(text: str) -> RuleSet:
    """Parse PAD-US rule configuration text without executing it."""
    return _Parser(text).parse()


def load_rule_config(path: Path) -> tuple[str, RuleSet]:
    """Read and parse a PAD-US rule configuration file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuleConfigError(f"Could not read PAD-US rules from {path}: {error}") from error
    try:
        return text, parse_rule_config(text)
    except RuleConfigError as error:
        raise RuleConfigError(f"Invalid PAD-US rules in {path}: {error}") from error


def build_rule_context(attributes: Mapping[str, object]) -> dict[str, object]:
    """Build the raw and virtual values referenced by configured rules."""
    context = {field: attributes.get(field) for field in PADUS_RULE_FIELDS}
    mang_name = attributes.get("Mang_Name")
    loc_mang = attributes.get("Loc_Mang")
    if loc_mang == NASA_LOCAL_MANAGER:
        manager = "NASA"
    elif mang_name == "DOE":
        manager = "DOE"
    else:
        manager = mang_name
    context[VIRTUAL_MANAGER_FIELD] = manager
    return context


def _tokenize(text: str) -> list[Token]:
    """Tokenize the rule DSL and reject all unsupported characters."""
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    tokens = []
    offset = 0
    for match in TOKEN_PATTERN.finditer(uncommented):
        skipped = uncommented[offset : match.start()]
        if skipped.strip():
            raise RuleConfigError(
                f"Unsupported text {skipped.strip()!r} near offset {offset}."
            )
        tokens.append(Token(match.group(), match.start()))
        offset = match.end()
    if uncommented[offset:].strip():
        raise RuleConfigError(
            f"Unsupported text {uncommented[offset:].strip()!r} near offset {offset}."
        )
    if not tokens:
        raise RuleConfigError("Rule configuration is empty.")
    return tokens


def _evaluate_expression(
    expression: Expression,
    attributes: Mapping[str, object],
    evaluate_rule: Callable[[str], bool],
) -> bool:
    """Evaluate one parsed expression node."""
    if isinstance(expression, Membership):
        matches = attributes.get(expression.field) in expression.values
        return not matches if expression.negate else matches
    if isinstance(expression, Reference):
        return evaluate_rule(expression.name)
    if expression.operator == "AND":
        return _evaluate_expression(
            expression.left, attributes, evaluate_rule
        ) and _evaluate_expression(expression.right, attributes, evaluate_rule)
    if expression.operator == "OR":
        return _evaluate_expression(
            expression.left, attributes, evaluate_rule
        ) or _evaluate_expression(expression.right, attributes, evaluate_rule)
    raise RuleConfigError(f"Unsupported Boolean operator: {expression.operator!r}.")


def _walk_expression(expression: Expression):
    """Yield an expression tree depth-first."""
    yield expression
    if isinstance(expression, BooleanExpression):
        yield from _walk_expression(expression.left)
        yield from _walk_expression(expression.right)
