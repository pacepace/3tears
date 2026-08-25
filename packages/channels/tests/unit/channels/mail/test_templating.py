"""What an email template is allowed to do, and what it must refuse.

The substitution surface is deliberately not a template language. Every value that
reaches it comes from a product's own data -- a respondent's name, a per-recipient
unsubscribe token -- and two of the three places it lands (the subject, the HTML part)
are injection targets. So the tests below are mostly about refusals: an unknown
placeholder, a newline in a value, an HTML part with no plain-text alternative.
"""

from __future__ import annotations

import pytest

from threetears.channels.mail.templating import EmailTemplate, TemplateRenderError


class TestSubstitution:
    def test_it_substitutes_into_subject_and_body(self) -> None:
        template = EmailTemplate(
            subject="Your {survey} invitation",
            body_text="Hello {name}, open {link}.",
        )

        message = template.render(to="ada@acme.example", values={"survey": "Q3", "name": "Ada", "link": "https://x/1"})

        assert message.subject == "Your Q3 invitation"
        assert message.body_text == "Hello Ada, open https://x/1."
        assert message.to == "ada@acme.example"

    def test_a_placeholder_with_no_value_is_refused_by_name(self) -> None:
        """Silently rendering `Hello , open` is the failure mode this exists to stop:
        it reaches a real inbox looking like a bug in the product, not in the send."""
        template = EmailTemplate(subject="s", body_text="Hello {name}")

        with pytest.raises(TemplateRenderError, match="name"):
            template.render(to="ada@acme.example", values={})

    def test_literal_braces_survive(self) -> None:
        template = EmailTemplate(subject="s", body_text="{{not a placeholder}} but {yes} is")

        message = template.render(to="ada@acme.example", values={"yes": "this"})

        assert message.body_text == "{not a placeholder} but this is"

    def test_a_value_is_never_itself_scanned_for_placeholders(self) -> None:
        """One pass, not a fixed point. A value carrying `{admin_token}` must land as
        those literal characters rather than being resolved against the value map."""
        template = EmailTemplate(subject="s", body_text="answer: {answer}")

        message = template.render(to="ada@acme.example", values={"answer": "{secret}", "secret": "leaked"})

        assert message.body_text == "answer: {secret}"

    def test_the_placeholders_it_needs_are_reportable(self) -> None:
        """So a product can validate an authored template against its own value set at
        authoring time rather than at send time."""
        template = EmailTemplate(subject="{a}", body_text="{b}", body_html="<p>{c}</p>")

        assert template.placeholders() == frozenset({"a", "b", "c"})


class TestInjectionRefusals:
    def test_a_newline_in_a_subject_value_is_refused(self) -> None:
        """A subject is a header. A value carrying CRLF appends headers of the caller's
        choosing -- a second `To`, a `Bcc`. The stdlib would refuse it later at fold
        time as an opaque send failure; refused here it names the placeholder."""
        template = EmailTemplate(subject="Invitation for {name}", body_text="hi")

        with pytest.raises(TemplateRenderError, match="name"):
            template.render(to="ada@acme.example", values={"name": "Ada\r\nBcc: mallory@evil.example"})

    def test_a_newline_in_a_body_value_is_allowed(self) -> None:
        """A body is not a header, and a multi-line value is ordinary there."""
        template = EmailTemplate(subject="s", body_text="{note}")

        message = template.render(to="ada@acme.example", values={"note": "line one\nline two"})

        assert message.body_text == "line one\nline two"

    def test_a_value_is_html_escaped_in_the_html_part_only(self) -> None:
        template = EmailTemplate(subject="s", body_text="hi {name}", body_html="<p>hi {name}</p>")

        message = template.render(to="ada@acme.example", values={"name": "<script>x</script>"})

        assert message.body_html == "<p>hi &lt;script&gt;x&lt;/script&gt;</p>"
        assert message.body_text == "hi <script>x</script>"

    def test_a_non_string_value_is_refused(self) -> None:
        """`None` renders as the word `None` into a live inbox, and an int renders in
        whatever `str` decides. Both are the product's bug and both are silent."""
        template = EmailTemplate(subject="s", body_text="{count}")

        with pytest.raises(TemplateRenderError, match="count"):
            template.render(to="ada@acme.example", values={"count": 3})  # type: ignore[dict-item]


class TestThePlainTextAlternative:
    def test_an_html_part_requires_a_text_part(self) -> None:
        """A recipient reading in plain text, and every spam filter scoring the message,
        both see the text part. An HTML-only send is a deliverability decision nobody
        made deliberately."""
        with pytest.raises(TemplateRenderError, match="plain-text"):
            EmailTemplate(subject="s", body_text="", body_html="<p>hi</p>")

    def test_html_and_text_are_carried_separately(self) -> None:
        template = EmailTemplate(subject="s", body_text="plain", body_html="<p>rich</p>")

        message = template.render(to="ada@acme.example", values={})

        assert message.body_text == "plain"
        assert message.body_html == "<p>rich</p>"


class TestUnsubscribe:
    def test_the_unsubscribe_link_is_per_recipient(self) -> None:
        """It carries a token, so it substitutes like anything else -- a template with
        one fixed unsubscribe URL for every recipient cannot honour an opt-out."""
        template = EmailTemplate(
            subject="s",
            body_text="hi",
            list_unsubscribe_url="https://acme.example/u/{token}",
        )

        message = template.render(to="ada@acme.example", values={"token": "abc"})

        assert message.list_unsubscribe_url == "https://acme.example/u/abc"

    def test_the_unsubscribe_mailto_is_carried_through(self) -> None:
        template = EmailTemplate(subject="s", body_text="hi", list_unsubscribe_mailto="unsub@acme.example")

        message = template.render(to="ada@acme.example", values={})

        assert message.list_unsubscribe_mailto == "unsub@acme.example"
